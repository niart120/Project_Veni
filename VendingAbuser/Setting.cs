using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using static VendingAbuser.Misc;

namespace VendingAbuser
{
    public enum Hardware
    {
        DS,
        DSLite,
        DSi,
        TDS
    }

    static class ROMExt
    {
        public static UInt32 BaseTimer0(this ROM rom)
        {
            UInt32[] timer0s = new[]{0xC79U, 0xC67U, 0x1102U, 0x10F5U};
            return timer0s[(int)rom];
        }
    }
    public enum ROM
    {
        B1,
        W1,
        B2,
        W2
    }
    public class Setting
    {
        [JsonInclude]
        public Hardware hw = Hardware.DS;
        [JsonInclude]
        public ROM rom = ROM.B1;
        [JsonInclude]
        public int N;
        [JsonInclude]
        public bool useGPU;
        [JsonInclude]
        public int begin;
        [JsonInclude]
        public int end;

        [JsonInclude]
        public DateTime beginDate;
        [JsonInclude]
        public DateTime endDate;

        [JsonInclude]
        public bool useKeyinput;
        [JsonInclude]
        public bool[] usekeys = new bool[12];
        [JsonInclude]
        public UInt32[] MAC;

        public Setting(Hardware hw, ROM rom, int N, bool useGPU, UInt32[] MAC, Int32 begin, Int32 end, bool useKeyinput, bool[] usekeys, DateTime beginDate, DateTime endDate) { 
            this.hw = hw;
            this.rom = rom;
            this.N = N;
            this.useGPU = useGPU;
            this.MAC = MAC;
            this.begin = begin;
            this.end = end;
            this.useKeyinput = useKeyinput;
            usekeys.CopyTo(this.usekeys, 0);
            this.beginDate = beginDate;
            this.endDate = endDate;
        }

        public static string GetConfigFilePath()
        {
            string appFilePath = System.AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
            return appFilePath + $"/config.json";
        }

        public static Setting ReadConfig()
        {
            // 設定ファイルのフルパスを取得
            string configFile = GetConfigFilePath();

            if (File.Exists(configFile) == false)
            {
                // 設定ファイルなし
                return null;
            }

            using (var reader = new StreamReader(configFile, Encoding.UTF8))
            {
                // 設定ファイル読み込み
                string buf = reader.ReadToEnd();
                Setting result = null;
                try
                {
                    result = JsonSerializer.Deserialize<Setting>(buf);
                }
                catch (JsonException jex) 
                {
                    return null;    
                }
                // デシリアライズして返す
                return result;
            }
        }

        public static void WriteConfig(Setting cfg)
        {
            // シリアライズ;
            var jsconfig = new JsonSerializerOptions
            {
                IncludeFields = true,
            };
            string buf = JsonSerializer.Serialize(cfg, jsconfig);

            // 設定ファイルのフルパス取得
            string configFile = GetConfigFilePath();

            using (var writer = new StreamWriter(configFile, false, Encoding.UTF8))
            {
                // 設定ファイルに書き込む
                writer.Write(buf);
            }
        }

    }
}
