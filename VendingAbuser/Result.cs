using static VendingAbuser.Misc;
namespace VendingAbuser
{
    public class Result : IComparable<Result>
    {
        private UInt64 initseed;
        private UInt32 datecode;
        private UInt32 timecode;
        private UInt32 timer0;
        private Int32 targetadv;
        private UInt32 keyinput;

        public Result(UInt64 initseed, UInt32 datecode, UInt32 timecode, UInt32 timer0, Int32 targetadv)
        {
            this.initseed = initseed;
            this.datecode = datecode;
            this.timecode = timecode;
            this.timer0 = timer0;
            this.targetadv = targetadv;
            this.keyinput = 0xFFFFFFFF;
        }
        public Result(UInt64 initseed, UInt32 datecode, UInt32 timecode, UInt32 timer0, Int32 targetadv, UInt32 keyinput)
        {
            this.initseed = initseed;
            this.datecode = datecode;
            this.timecode = timecode;
            this.timer0 = timer0;
            this.targetadv = targetadv;
            this.keyinput = keyinput;
        }

        public string[] Convert()
        {
            var date = DatecodeToYYMMDD(datecode);
            var time = TimecodeToHHMMSS(timecode);
            var datetime = new DateTime(2000 + (int)date[0], (int)date[1], (int)date[2], (int)time[0], (int)time[1], (int)time[2]);

            var seed_ = initseed.ToString("X");
            var date_ = datetime.ToString("d");
            var time_ = datetime.ToString("T");
            var timer0_ = timer0.ToString("X");
            var targetadv_ = targetadv.ToString();
            var keyinput_ = keyinput == 0xFFFFFFFF ? "" : KeycodeToKeyinput(this.keyinput);
            return new[] { seed_, date_, time_, timer0_, targetadv_, keyinput_ };
        }
        override public string ToString()
        {
            var date = DatecodeToYYMMDD(datecode);
            var time = TimecodeToHHMMSS(timecode);
            var datetime = new DateTime(2000 + (int)date[0], (int)date[1], (int)date[2], (int)time[0], (int)time[1], (int)time[2]);

            var seed_ = "0x" + initseed.ToString("X");
            var date_ = datetime.ToString("d");
            var time_ = datetime.ToString("T");
            var timer0_ = timer0.ToString("X");
            var targetadv_ = targetadv.ToString();
            var keyinput_ = keyinput == 0xFFFFFFFF ? "" : ", " + KeycodeToKeyinput(this.keyinput);
            return string.Join(", ", this.Convert()[0..^1]) + keyinput_;
        }

        public int CompareTo(Result? other)
        {
            if (other is null) return 1;
            var dt = ((UInt64)timecode) << 32 ^ datecode;
            var otherdt = ((UInt64)other.timecode) << 32 ^ other.datecode;
            return Misc.ChangeEndian(dt).CompareTo(Misc.ChangeEndian(otherdt));
        }

        public override bool Equals(object? obj)
        {
            if (!(obj is Result)) return false;
            if (obj == null) return false;
            var other = (Result)obj;
            if (CompareTo(other) != 0) return false;
            return this.initseed == other.initseed;
        }

        public override int GetHashCode()
        {
            return (initseed, datecode, timecode).GetHashCode();
        }
    }
}
