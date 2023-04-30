using VendingAbuser;
using System.Runtime.InteropServices;

namespace Test
{
    [TestClass]
    public class InitSeedGeneratorTest
    {
        public byte[] InitialMessageGeneratorB(uint[] mac, uint t0, UInt32 dateCode, UInt32 timeCode)
        {
            UInt32[] nazo = new UInt32[5];
            nazo[0] = 0x02215F10;
            nazo[1] = nazo[0] + 0xFC;
            nazo[2] = nazo[0] + 0xFC;
            nazo[3] = nazo[0] + 0x148;
            nazo[4] = nazo[0] + 0x148;

            UInt32 v = 0x60;
            UInt32 frame = 8;

            var W = new uint[13];
            W[0] = ChangeEndian(nazo[0]);
            W[1] = ChangeEndian(nazo[1]);
            W[2] = ChangeEndian(nazo[2]);
            W[3] = ChangeEndian(nazo[3]);
            W[4] = ChangeEndian(nazo[4]);

            W[5] = ChangeEndian((v << 16) | t0);
            W[6] = ChangeEndian((mac[4] << 16) | (mac[5]<<24));
            W[7] = ChangeEndian(0x6000000 ^ frame ^ (mac[3] << 24) | (mac[2] << 16) | (mac[1] << 8) | (mac[0]));
            W[8] = ChangeEndian(dateCode);
            W[9] = ChangeEndian(timeCode);
            W[10] = 0x00000000;
            W[11] = 0x00000000;
            W[12] = ChangeEndian(0x00002FFF);

            byte[] message = new byte[52];
            //Span<T> を使ったメッセージ作成
            var u32span = MemoryMarshal.Cast<byte, UInt32>(message);
            for (int i = 0;i<W.Length;i++)
            {
                u32span[i] = W[i];
            }
            return message;
        }

        public byte[] InitialMessageGeneratorW(uint[] mac, uint t0, UInt32 dateCode, UInt32 timeCode)
        {
            UInt32[] nazo = new UInt32[5];
            nazo[0] = 0x02215F30;
            nazo[1] = nazo[0] + 0xFC;
            nazo[2] = nazo[0] + 0xFC;
            nazo[3] = nazo[0] + 0x148;
            nazo[4] = nazo[0] + 0x148;

            UInt32 v = 0x5F;
            UInt32 frame = 8;

            var W = new uint[13];
            W[0] = ChangeEndian(nazo[0]);
            W[1] = ChangeEndian(nazo[1]);
            W[2] = ChangeEndian(nazo[2]);
            W[3] = ChangeEndian(nazo[3]);
            W[4] = ChangeEndian(nazo[4]);

            W[5] = ChangeEndian((v << 16) | t0);
            W[6] = ChangeEndian((mac[4] << 16) | (mac[5] << 24));
            W[7] = ChangeEndian(0x6000000 ^ frame ^ (mac[3] << 24) | (mac[2] << 16) | (mac[1] << 8) | (mac[0]));
            W[8] = ChangeEndian(dateCode);
            W[9] = ChangeEndian(timeCode);
            W[10] = 0x00000000;
            W[11] = 0x00000000;
            W[12] = ChangeEndian(0x00002FFF);

            byte[] message = new byte[52];
            //Span<T> を使ったメッセージ作成
            var u32span = MemoryMarshal.Cast<byte, UInt32>(message);
            for (int i = 0; i < W.Length; i++)
            {
                u32span[i] = W[i];
            }
            return message;
        }

        private UInt32 ChangeEndian(UInt32 val)
        {
            val = ((val << 8) & 0xFF00FF00U) | ((val >> 8) & 0xFF00FFU);
            return (val << 16) | (val >> 16);
        }

        public ulong Generate(byte[] message)
        {
            uint[] W = new uint[80];
            var u8span = MemoryMarshal.Cast<UInt32, byte>(W);
            for (int i = 0; i < message.Length; i++) u8span[i] = message[i];
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
            //Console.WriteLine((A+H0).ToString("X")+(B+H1).ToString("X")+(C+H2).ToString("X")+(D+H3).ToString("X")+(E+H4).ToString("X"));

            ulong seed = ChangeEndian(H1 + B);
            seed <<= 32;
            seed |= ChangeEndian(H0 + A);

            return seed;
        }

        [TestMethod]
        public void YatsunaTest()
        {
            var useGPU = true;
            var useKeyinput = false;
            var usekeys = new bool[] { false, false, false, false, false, false, false, false, false, false, false, false };
            var begin = 60;
            var end = 120;
            DateTime beginDate = new DateTime(2023, 1, 1);
            DateTime endDate = new DateTime(2023, 12, 31);
            Setting setting = new Setting(Hardware.DS, ROM.B1, 6, useGPU, new uint[] { 0, 0, 0, 0, 0, 0 }, begin, end, useKeyinput, usekeys, beginDate, endDate);

            InitSeedGenerator isg = new InitSeedGenerator(setting);
            UInt16 timer0 = 0xC79;//black
            UInt32 nnddmmyy = 0x06010100; //2000/01/01(Sat)
            UInt32 zzssmmhh = 0x00452301; //01:23:45(24h)(note:after12:00:00 then hh += 0x40
            UInt32 keyinput = 0x00002fff;//no input

            var msg = isg.GenerateMessage(timer0, nnddmmyy, zzssmmhh, keyinput);
            var seed = isg.Generate(msg);

            var expected = Generate(InitialMessageGeneratorB(new uint[] { 0, 0, 0, 0, 0, 0 }, timer0, nnddmmyy, zzssmmhh));

            Assert.AreEqual(expected, seed);

        }

        [TestMethod]
        public void ZeroValueTest()
        {
            var useGPU = true;
            var useKeyinput = false;
            var usekeys = new bool[] { false, false, false, false, false, false, false, false, false, false, false, false };
            var begin = 60;
            var end = 120;
            DateTime beginDate = new DateTime(2023, 1, 1);
            DateTime endDate = new DateTime(2023, 12, 31);

            Setting setting = new Setting(Hardware.DS, ROM.B1, 6, useGPU, new uint[] { 0, 0, 0, 0, 0, 0 }, begin, end, useKeyinput, usekeys, beginDate, endDate);
            InitSeedGenerator isg = new InitSeedGenerator(setting);
            UInt16 timer0 = 0xC79;//black
            UInt32 nnddmmyy = 0x06010100; //2000/01/01(Sat)
            UInt32 zzssmmhh = 0x00000000; //00:00:00(24h)(note:after12:00:00 then hh += 0x40
            UInt32 keyinput = 0x00002fff;//no input

            var msg = isg.GenerateMessage(timer0, nnddmmyy, zzssmmhh, keyinput);
            var seed = isg.Generate(msg);

            var add = 0x269ec3UL;
            var mul = 0x5d588b656c078965UL;
            
            seed = seed * mul + add;

            Assert.AreEqual(0xF90E36420FDE1271UL, seed);
        }

        [TestMethod]
        public void ActualDateTest()
        {
            var useGPU = true;
            var useKeyinput = false;
            var usekeys = new bool[] { false, false, false, false, false, false, false, false, false, false, false, false };
            var begin = 60;
            var end = 120;
            DateTime beginDate = new DateTime(2023, 1, 1);
            DateTime endDate = new DateTime(2023, 12, 31);

            Setting setting = new Setting(Hardware.DS, ROM.B1, 6, useGPU, new uint[] { 0, 0, 0, 0, 0, 0 }, begin, end, useKeyinput, usekeys, beginDate, endDate);
            InitSeedGenerator isg = new InitSeedGenerator(setting);
            UInt32 timer0 = 0xC79;//black
            UInt32 nnddmmyy = 0x01100423; //2023/04/10(Mon)
            UInt32 zzssmmhh = 0x00452301; //01:23:45(24h)(note:after12:00:00 then hh += 0x40
            UInt32 keyinput = 0x00002fff;//no input

            var add = 0x269ec3UL;
            var mul = 0x5d588b656c078965UL;

            var msg = isg.GenerateMessage(timer0, nnddmmyy, zzssmmhh, keyinput);
            var seed = isg.Generate(msg);
            seed = seed * mul + add;

            Assert.AreEqual(0x35F6299857B2A22DU, seed);
        }
        [TestMethod]
        public void ActualMACTest()
        {
            var useGPU = true;
            var mac = new uint[] { 0x00, 0x1a, 0xe9, 0x03, 0x56, 0xbe };
            
            var useKeyinput = false;
            var usekeys = new bool[] { false, false, false, false, false, false, false, false, false, false, false, false };
            var begin = 60;
            var end = 120;
            DateTime beginDate = new DateTime(2023, 1, 1);
            DateTime endDate = new DateTime(2023, 12, 31);

            Setting setting = new Setting(Hardware.DSLite, ROM.W1, 6, useGPU, mac, begin, end, useKeyinput, usekeys, beginDate, endDate);
            InitSeedGenerator isg = new InitSeedGenerator(setting);
            UInt32 timer0 = 0xC68;//white
            UInt32 nnddmmyy = 0x01100423; //2023/04/10(Mon)
            UInt32 zzssmmhh = 0x00452301; //01:23:45(24h)(note:after12:00:00 then hh += 0x40
            UInt32 keyinput = 0x00002fff;//no input

            var add = 0x269ec3UL;
            var mul = 0x5d588b656c078965UL;

            var msg = isg.GenerateMessage(timer0, nnddmmyy, zzssmmhh, keyinput);
            var seed = isg.Generate(msg);

            //var yatsuna = Generate(InitialMessageGeneratorW(mac, timer0, nnddmmyy, zzssmmhh));
            //yatsuna = yatsuna * mul + add;
            seed = seed * mul + add;
            //Console.WriteLine(seed.ToString("X"));
            //Console.WriteLine(yatsuna.ToString("X"));
            Assert.AreEqual(0x128AF3A96C35C3A2U, seed);
        }

    }
}