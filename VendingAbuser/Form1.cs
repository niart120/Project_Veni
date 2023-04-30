using System.Text;

namespace VendingAbuser
{
    public partial class Form1 : Form
    {
        private CancellationTokenSource cts;
        private bool[] usekeys = new bool[] { false, false, false, false, false, false, false, false, false, false, false, false };
        public Form1()
        {
            InitializeComponent();
            this.comboBox1.SelectedIndex = 0;
            this.comboBox2.SelectedIndex = 0;
            this.FormClosing += new FormClosingEventHandler(Form1_FormClosing);
            this.ResumeLayout(false);
        }

        private IEnumerable<T> WrapForGUI<T>(IEnumerable<T> source, CancellationToken ct)
        {
            foreach(var item in source)
            {
                progressBar1.BeginInvoke(() =>
                {
                    progressBar1.Increment(1);
                });
                yield return item;
                if (ct.IsCancellationRequested) break;

            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Form2 form2 = new Form2();
            DialogResult dr = form2.ShowDialog(usekeys);
            if (dr == DialogResult.OK)
            {
                form2.GetValues().CopyTo(usekeys, 0);
            }
        }

        private async void button2_Click(object sender, EventArgs e)
        {
            button2.Enabled = false;
            button3.Enabled = true;

            cts = new CancellationTokenSource();
            var cfg = GetSetting();
            var chunksize = 2_500_000;
            var paramlength = Misc.GetParamsLength(cfg);

            this.progressBar1.Maximum = (int)((paramlength)/chunksize) ;
            this.progressBar1.Value = 0;

            var chunk = WrapForGUI(Misc.GetParams(cfg).Chunk(chunksize), cts.Token);

            await Task.Run(() =>
            {
                var lst = new List<Result>();
                foreach (var messageParams in chunk)
                {
                    var tmp = InitSeedSearch.BruteForceSearch(cfg, messageParams);
                    lst.AddRange(tmp);
                }
                var results = lst.Order().ToArray();
                Invoke((MethodInvoker)(() =>
                {
                    this.dataGridView1.Rows.Clear();
                    foreach (var result in results)
                    {
                        this.dataGridView1.Rows.Add(result.Convert());
                    }
                    SaveResult(results, cfg.useKeyinput);
                    button2.Enabled = true;
                    button3.Enabled = false;
                }));
            });
        }

        private void button3_Click(object sender, EventArgs e)
        {
            cts.Cancel();
            button3.Enabled = false;
        }

        private void Form1_FormClosing(object? sender, FormClosingEventArgs e)
        {
            SaveToSetting();
        }

        private Setting GetSetting()
        {
            var hw = (Hardware)this.comboBox1.SelectedIndex;
            var rom = (ROM)this.comboBox2.SelectedIndex;
            var N = (int)this.numericUpDown1.Value;
            var useGPU = this.checkBox1.Checked;
            var mac = new UInt32[] { (UInt32)this.numericUpDown2.Value, (UInt32)this.numericUpDown3.Value, (UInt32)this.numericUpDown4.Value, (UInt32)this.numericUpDown5.Value, (UInt32)this.numericUpDown6.Value, (UInt32)this.numericUpDown7.Value };
            var begin = (int)this.numericUpDown8.Value;
            var end = (int)this.numericUpDown9.Value;
            var useKeyinput = this.checkBox2.Checked;
            var begindate = this.dateTimePicker1.Value;
            var enddate = this.dateTimePicker2.Value;

            var cfg = new Setting(hw, rom, N, useGPU, mac, begin, end, useKeyinput, usekeys, begindate, enddate);
            return cfg;
        }

        public static string GetResultFilePath()
        {
            string appFilePath = System.AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
            return appFilePath + $"/result.txt";
        }

        public static void SaveResult(Result[] results, bool useKeyinput)
        {
            if (results.Length == 0) return;
            string resultFile = GetResultFilePath();
            using (var writer = new StreamWriter(resultFile, false, Encoding.UTF8))
            {
                //seed_, date_, time_, timer0_, targetadv_, keyinput_
                writer.WriteLine("èâä˙seed, ì˙ït, éûçè, timer0, è¡îÔêî"+ (useKeyinput? ", ÉLÅ[ì¸óÕ":""));
                foreach(var result in results)
                {
                    writer.WriteLine(result);
                }
            }
        }

        private void SaveToSetting()
        {
            Setting.WriteConfig(GetSetting());
        }

        private void LoadFromSetting()
        {
            var cfg = Setting.ReadConfig();
            if (cfg == null)
            {
                return;
            }

            this.comboBox1.SelectedIndex = (int)cfg.hw;
            this.comboBox2.SelectedIndex = (int)cfg.rom;

            this.numericUpDown1.Value = cfg.N;

            this.checkBox1.Checked = cfg.useGPU;

            this.numericUpDown2.Value = cfg.MAC[0];
            this.numericUpDown3.Value = cfg.MAC[1];
            this.numericUpDown4.Value = cfg.MAC[2];
            this.numericUpDown5.Value = cfg.MAC[3];
            this.numericUpDown6.Value = cfg.MAC[4];
            this.numericUpDown7.Value = cfg.MAC[5];

            this.numericUpDown8.Value = cfg.begin;
            this.numericUpDown9.Value = cfg.end;

            this.checkBox2.Checked = cfg.useKeyinput;
            cfg.usekeys.CopyTo(this.usekeys, 0);

        }

        private void numericUpDown2_Enter(object sender, EventArgs e)
        {
            this.numericUpDown2.Select(0, numericUpDown2.Text.Length);
        }
        private void numericUpDown3_Enter(object sender, EventArgs e)
        {
            this.numericUpDown3.Select(0, numericUpDown3.Text.Length);
        }
        private void numericUpDown4_Enter(object sender, EventArgs e)
        {
            this.numericUpDown4.Select(0, numericUpDown4.Text.Length);
        }
        private void numericUpDown5_Enter(object sender, EventArgs e)
        {
            this.numericUpDown5.Select(0, numericUpDown5.Text.Length);
        }
        private void numericUpDown6_Enter(object sender, EventArgs e)
        {
            this.numericUpDown6.Select(0, numericUpDown6.Text.Length);
        }
        private void numericUpDown7_Enter(object sender, EventArgs e)
        {
            this.numericUpDown7.Select(0, numericUpDown7.Text.Length);
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            LoadFromSetting();
        }
    }
}