using System.Runtime.CompilerServices;

namespace VendingAbuser
{
    public static class Misc
    {
        private static readonly UInt32[] BCD = new UInt32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153 };

        public static UInt32 DecToBCD(UInt32 val)
        {
            return val + val / 10 * 6;
        }

        public static UInt32 BCDToDec(UInt32 val)
        {
            return (val >> 4) * 10 + (val & 0xF);
        }

        public static UInt32 YYMMDDToDatecode(UInt32[] val)
        {
            //algorithm:https://en.wikipedia.org/wiki/Zeller%27s_congruence
            var year = 2000U + val[0];
            var m = val[1];
            var d = val[2];
            if (m == 1 || m == 2)
            {
                m += 12;
                year--;
            }
            var y = year % 100;
            var j = year / 100;

            var nn = (d + 13 * (m + 1) / 5 + y + y / 4 + j / 4 - 2 * j + 6) % 7;

            //nnddmmyy
            var result = nn;
            result <<= 8;
            result |= DecToBCD(val[2]);
            result <<= 8;
            result |= DecToBCD(val[1]);
            result <<= 8;
            result |= DecToBCD(val[0]);

            return result;
        }

        public static UInt32 HHMMSSToTimecode(UInt32[] val)
        {
            //zzssmmhh
            var result = DecToBCD(val[2]);
            result <<= 8;
            result |= DecToBCD(val[1]);
            result <<= 8;
            result |= DecToBCD(val[0]);
            if (val[0] >= 12) result += 0x40;

            return result;
        }

        public static UInt32[] DatecodeToYYMMDD(UInt32 datecode)
        {
            var result = new UInt32[3];
            result[0] = BCDToDec(datecode & 0xFF);
            datecode >>= 8;
            result[1] = BCDToDec(datecode & 0xFF);
            datecode >>= 8;
            result[2] = BCDToDec(datecode & 0xFF);
            return result;
        }

        public static UInt32[] TimecodeToHHMMSS(UInt32 timecode)
        {
            var result = new UInt32[3];
            if ((timecode & 0xFF) >= 0x40) timecode -= 0x40;
            result[0] = BCDToDec(timecode & 0xFF);
            timecode >>= 8;
            result[1] = BCDToDec(timecode & 0xFF);
            timecode >>= 8;
            result[2] = BCDToDec(timecode & 0xFF);

            return result;
        }

        public static string KeycodeToKeyinput(UInt32 keycode)
        {
            string[] keystrings = {"A","B","St", "Sl", "→", "←", "↑", "↓", "R", "L", "X", "Y"};
            var keys = ~(keycode & 0xFFF);
            var result = "";
            for(int i=0; i < keystrings.Length; i++)
            {
                var t = 1<<i;
                if((t&keys)==t) result += keystrings[i];
            }
            return result;
        }

        public static UInt32[] GenerateKeycodeList(Setting cfg)
        {
            var result = new List<UInt32>();
            if (cfg.useKeyinput == false)
            {
                result.Add(0x2fff);
                return result.ToArray();
            }
            var keys = 0U;
            foreach (var key in cfg.usekeys)
            {
                keys <<= 1;
                keys ^= Convert.ToUInt32(key);
            }

            const UInt32 LeftRight = 0x030U;
            const UInt32 UpDown = 0x0C0U;
            const UInt32 StSlLR = 0x30CU;
            
            for (UInt32 i = 0U; i < (1 << 12); ++i)
            {
                if ((i & keys) == i)
                {
                    if ((i & LeftRight) == LeftRight) continue;
                    if ((i & UpDown) == UpDown) continue;
                    if ((i & StSlLR) == StSlLR) continue;
                    result.Add(i);
                }
            }

            return result.Select(x => x ^ 0x2FFFU).ToArray();
        }

        private static int Size(this Range range)
        {
            return range.End.Value - range.Start.Value;
        }

        public static UInt32[] GenerateDatecodeList(DateTime beginDate, DateTime endDate)
        {
            beginDate = new DateTime(beginDate.Year, beginDate.Month, beginDate.Day, 0,0,0);
            endDate = new DateTime(endDate.Year, endDate.Month, endDate.Day, 23, 59, 59);
            if (beginDate>endDate) throw new ArgumentOutOfRangeException();

            var result = new List<UInt32>();

            for (var date = beginDate; date <= endDate; date = date.AddDays(1))
            {
                var yy = BCD[date.Year % 100];
                var mm = BCD[date.Month];
                var dd = BCD[date.Day];
                var nn = (uint)date.DayOfWeek;
                var datecode = (nn << 24) ^ (dd << 16) ^ (mm << 8) ^ yy;
                result.Add(datecode);
            }
            return result.ToArray();
        }

        public static UInt32[] GenerateDatecodeList(Range yrange, Range mrange, Range drange)
        {
            if (yrange.Size() * mrange.Size() * drange.Size() == 0) throw new ArgumentOutOfRangeException();

            var result = new List<UInt32>();
            var nn = 1U;//曜日
            var lastdaycheck = new HashSet<UInt32>() { 0x4, 0x6, 0x9, 0x11 };

            var yset = new HashSet<UInt32>(BCD[yrange]);
            var mset = new HashSet<UInt32>(BCD[mrange]);
            var dset = new HashSet<UInt32>(BCD[drange]);

            foreach (var yy in BCD[0..100])
            {
                foreach (var mm in BCD[1..13])
                {
                    var lastday = 31;
                    if (lastdaycheck.Contains(mm))
                    {
                        lastday = 30;
                    }
                    else if (mm == 2)
                    {
                        lastday = 28;
                        if (BCDToDec(yy) % 4 == 0)
                        {
                            lastday = 29;
                        }
                    }
                    foreach (var dd in BCD[1..(lastday + 1)])
                    {
                        //nnddmmyy
                        var datecode = (nn << 24) ^ (dd << 16) ^ (mm << 8) ^ yy;
                        if (yset.Contains(yy) && mset.Contains(mm) && dset.Contains(dd)) result.Add(datecode);
                        ++nn;
                        nn %= 7;
                    }
                }
            }
            return result.ToArray();
        }

        public static UInt32[] GenerateDatecodeList()
        {
            var st = new DateTime(2000, 1, 1);
            var ed = new DateTime(2099, 12, 31);
            return GenerateDatecodeList(st, ed);
        }

        public static UInt32[] GenerateTimecodeList(Range hrange, Range mrange, Range srange)
        {
            if (hrange.Size() * mrange.Size() * srange.Size() == 0) throw new ArgumentOutOfRangeException();
            var result = new List<UInt32>();
            foreach (var h in BCD[hrange])
            {
                var hh = h;
                if (h >= 0x12) hh += 0x40;
                foreach (var mm in BCD[mrange])
                {
                    foreach (var ss in BCD[srange])
                    {
                        //zzssmmhh
                        var timecode = (ss << 16) ^ (mm << 8) ^ hh;
                        result.Add(timecode);
                    }
                }
            }
            return result.ToArray();
        }
        public static UInt32[] GenerateTimecodeList()
        {
            return GenerateTimecodeList(0..24, 0..60, 0..60);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 ChangeEndian(UInt32 val)
        {
            val = ((val << 8) & 0xFF00FF00U) | ((val >> 8) & 0xFF00FFU);
            return (val << 16) | (val >> 16);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 ChangeEndian(UInt64 val)
        {
            var upper = (UInt32)(val >> 32);
            var lower = (UInt32)(val & 0xFFFFFFFFU);
            return ((UInt64)ChangeEndian(lower)<<32) | ChangeEndian(upper);
        }

        public static long GetParamsLength(Setting cfg)
        {
            var t0range = (cfg.rom == ROM.B1 || cfg.rom == ROM.W1) ? 3L : 7L;
            var kirange = (long)GenerateKeycodeList(cfg).Length;
            var dcrange = (long)GenerateDatecodeList(cfg.beginDate, cfg.endDate).Length;
            var tcrange = (long)GenerateTimecodeList().Length;
            return t0range * kirange * dcrange * tcrange;
        }


        public static IEnumerable<(uint, uint, uint, uint)> GetParams(Setting cfg, UInt32[] datecodeList, UInt32[] timecodeList)
        {
            UInt32[] keycodeList = GenerateKeycodeList(cfg);
            //timer0の範囲
            var t0range = (cfg.rom == ROM.B1 || cfg.rom == ROM.W1) ? 3L : 7L;
            var t = cfg.rom.BaseTimer0() + t0range;
            for (UInt32 timer0 = cfg.rom.BaseTimer0(); timer0 < t; ++timer0)
            {
                foreach (var keyinput in keycodeList)
                {
                    foreach (var datecode in datecodeList)
                    {
                        foreach (var timecode in timecodeList)
                        {
                            yield return (timer0, datecode, timecode, keyinput);
                        }
                    }
                }
            }
        }

        public static IEnumerable<(uint, uint, uint, uint)> GetParams(Setting cfg)
        {
            UInt32[] datecodeList = GenerateDatecodeList(cfg.beginDate, cfg.endDate);
            UInt32[] timecodeList = GenerateTimecodeList();
            return GetParams(cfg,  datecodeList, timecodeList);
        }

        public static IEnumerable<IEnumerable<T>> Chunk<T>(this IEnumerable<T> source, int chunkSize)
        {
            if (chunkSize <= 0)
                throw new ArgumentException("Chunk size must be greater than 0.", nameof(chunkSize));

            while (source.Any())
            {
                yield return source.Take(chunkSize);
                source = source.Skip(chunkSize);
            }
        }
    }
}
