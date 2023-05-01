using CUDASearch;
using static VendingAbuser.Misc;

namespace VendingAbuser
{
    public class InitSeedGenerator
    {
        private readonly UInt32[] baseMessage;
        public InitSeedGenerator(Setting cfg)
        {
            UInt32[] message = new UInt32[13];
            UInt32[] nazo = new UInt32[5];
            UInt32 vcount = 0U;
            if (cfg.rom == ROM.B1)
            {
                nazo[0] = 0x02215F10;
                vcount = 0x60;
                //timer0:0xC79~0xC7A
            }
            if (cfg.rom == ROM.W1)
            {
                nazo[0] = 0x02215F30;
                vcount = 0x5F;
                //timer0:0xC67~0xC69
            }
            if (cfg.rom == ROM.B2)
            {
                nazo[0] = 0x0209A8DC;
                nazo[1] = 0x02039AC9;
                nazo[2] = 0x021FF9B0;
                vcount = 0x82;
            }

            if (cfg.rom == ROM.W2)
            {
                nazo[0] = 0x0209A8FC;
                nazo[1] = 0x02039AF5;
                nazo[2] = 0x021FF9D0;
                vcount = 0x82;
            }

            if (cfg.rom == ROM.B1 || cfg.rom == ROM.W1)
            {
                nazo[1] = nazo[0] + 0xFC;
                nazo[2] = nazo[0] + 0xFC;
                nazo[3] = nazo[0] + 0x148;
                nazo[4] = nazo[0] + 0x148;
            }
            else
            {
                nazo[3] = nazo[2] + 0x54;
                nazo[4] = nazo[2] + 0x54;
            }

            UInt32 gxstat = 0x06000000U;
            UInt32 frame;

            if (cfg.hw == Hardware.DS)
            {
                frame = 0x8;
            }
            else if (cfg.hw == Hardware.DSLite)
            {
                frame = 0x6;
            }
            else
            {
                frame = 0x9;
            }

            UInt32 lowmac = (cfg.MAC[4] << 16) | (cfg.MAC[5]<<24);

            UInt32 mac_gxstat_frame = (cfg.MAC[0]) ^ (cfg.MAC[1] << 8) ^ (cfg.MAC[2] << 16) ^ (cfg.MAC[3] << 24) ^ gxstat ^ frame;


            //Span<T> を使ったメッセージ作成

            for (int i = 0; i < nazo.Length; i++)
            {
                message[i] = nazo[i];
            }
            message[5] = (vcount << 16);
            message[6] = lowmac;
            message[7] = mac_gxstat_frame;
            this.baseMessage = message;
        }

        public UInt32[] GenerateMessage(UInt32 timer0, UInt32 nnddmmyy, UInt32 ssmihh, UInt32 keyinput)
        {
            UInt32[] msg = new UInt32[13];
            this.baseMessage.AsSpan().CopyTo(msg);
            msg[5] ^= timer0;
            msg[8] = nnddmmyy;
            msg[9] = ssmihh;
            msg[12] = keyinput;
            return msg;
        }

        public UInt32[] GetBaseMessage()
        {
            return baseMessage;
        }

        public UInt64 Generate(UInt32[] msg)
        {
            //original author:@sub_yatsuna[https://gist.github.com/yatsuna827/628138975c86123bdfdf9ba98001c613]
            uint[] W = new uint[80];
            for(int i=0; i < msg.Length; i++)
            {
                W[i] = ChangeEndian(msg[i]);
            }
            
            W[13] = 0x80000000U;
            W[14] = 0x00000000U;
            W[15] = 0x000001A0U;


            uint t;
            for (t = 16; t < 80; t++)
            {
                var w = W[t - 3] ^ W[t - 8] ^ W[t - 14] ^ W[t - 16];
                W[t] = (w << 1) | (w >> 31);
            }

            const uint H0 = 0x67452301;
            const uint H1 = 0xEFCDAB89;
            const uint H2 = 0x98BADCFE;
            const uint H3 = 0x10325476;
            const uint H4 = 0xC3D2E1F0;

            uint A, B, C, D, E;
            A = H0; B = H1; C = H2; D = H3; E = H4;

            for (t = 0; t < 20; t++)
            {
                var temp = ((A << 5) | (A >> 27)) + ((B & C) | ((~B) & D)) + E + W[t] + 0x5A827999;
                E = D;
                D = C;
                C = (B << 30) | (B >> 2);
                B = A;
                A = temp;
            }
            for (; t < 40; t++)
            {
                var temp = ((A << 5) | (A >> 27)) + (B ^ C ^ D) + E + W[t] + 0x6ED9EBA1;
                E = D;
                D = C;
                C = (B << 30) | (B >> 2);
                B = A;
                A = temp;
            }
            for (; t < 60; t++)
            {
                var temp = ((A << 5) | (A >> 27)) + ((B & C) | (B & D) | (C & D)) + E + W[t] + 0x8F1BBCDC;
                E = D;
                D = C;
                C = (B << 30) | (B >> 2);
                B = A;
                A = temp;
            }
            for (; t < 80; t++)
            {
                var temp = ((A << 5) | (A >> 27)) + (B ^ C ^ D) + E + W[t] + 0xCA62C1D6;
                E = D;
                D = C;
                C = (B << 30) | (B >> 2);
                B = A;
                A = temp;
            }

            ulong seed = ChangeEndian(H1 + B);
            seed <<= 32;
            seed |= ChangeEndian(H0 + A);

            return seed;
        }
    }

    public class InitSeedSearch
    {
        const UInt64 mul = 0x5d588b656c078965UL;
        const UInt64 add = 0x269ec3UL;

        public static IEnumerable<Result> BruteForceSearch(Setting cfg, IEnumerable<(UInt32, UInt32, UInt32, UInt32)> messageParams)
        {
            var parameters = messageParams.ToArray();
            var length = parameters.Length;
            var isg = new InitSeedGenerator(cfg);
            var vms = new VendingMachineSearcher(cfg);

            const UInt64 mul = 0x5d588b656c078965UL;
            const UInt64 add = 0x269ec3UL;
            UInt64 initmul = 1;
            UInt64 initadd = 0;
            UInt64 tmp = 1;
            for (int i = 0; i < cfg.begin; ++i)
            {
                initmul *= mul;
                initadd += tmp;
                tmp *= mul;
            }
            initadd *= add;

            var initseeds = new UInt64[length];
            var seeds = new UInt64[length];
            var bmsg = isg.GetBaseMessage();

            SeedGenerator sg = new SeedGenerator(initmul, initadd);

            if (cfg.useGPU)
            {
                UInt32[] messages = new UInt32[4 * length];
                Parallel.For(0, length, i =>
                {
                    (var timer0, var datecode, var timecode, var keycode) = parameters[i];
                    var pos = 4 * i;
                    messages[pos] = timer0;
                    messages[pos + 1] = datecode;
                    messages[pos + 2] = timecode;
                    messages[pos + 3] = keycode;
                });
                sg.calculate(bmsg, messages, initseeds, seeds, length);
            }
            else
            {
                Parallel.For(0, length, i =>
                {
                    (var timer0, var datecode, var timecode, var keycode) = parameters[i];
                    var msg = isg.GenerateMessage(timer0, datecode, timecode, keycode);
                    var initseed = isg.Generate(msg) * mul + add;
                    initseeds[i] = initseed;
                    seeds[i] = initseed * initmul + initadd;
                });
            }

            vms.calculate(seeds);//result:seeds[i] = i+1 (satisfied) | 0 (otherwise), inplace処理
            var idx_ = seeds[0..length].Max();
            while (idx_ > 0)
            {
                var initseed = initseeds[idx_ - 1];
                var (timer0, datecode, timecode, keyinput) = parameters[idx_ - 1];
                var advances = CalculateAdvance(initseed, cfg.N, cfg.begin, cfg.end);
                foreach (var adv in advances)
                {
                    var result = new Result(initseed, datecode, timecode, timer0, adv, cfg.useKeyinput ? keyinput : 0xFFFFFFFFU);
                    yield return result;
                }
                seeds[idx_ - 1] = 0UL;
                idx_ = seeds[0..length].Max();
            }
        }


        private static Int32[] CalculateAdvance(ulong initseed, int n, int begin, int end)
        {
            List<Int32> result = new List<Int32>();

            UInt64 state = initseed;
            UInt64 checkcode = 0xFFFFFFFFFFFFFFFFUL;
            UInt64 mask = (1UL << (5 * n)) - 1UL;

            int i = 0;
            for (; i < begin; ++i)
            {
                checkcode <<= 5;
                checkcode ^= (state >> 32) >> 27;
                state = state * mul + add;
            }
            for (; i < end; ++i)
            {
                checkcode <<= 5;
                checkcode ^= (state >> 32) >> 27;
                if((checkcode&mask)==0UL){
                    result.Add(i);
                };
                state = state * mul + add;
            }
            return result.ToArray();
        }
    }
}
