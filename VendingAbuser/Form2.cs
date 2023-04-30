using VendingAbuser;
namespace VendingAbuser
{
    public partial class Form2 : Form
    {
        public Form2()
        {
            InitializeComponent();
        }

        public DialogResult ShowDialog(bool[] usekeys)
        {
            checkBox1.Checked = usekeys[0];
            checkBox2.Checked = usekeys[1];
            checkBox3.Checked = usekeys[2];
            checkBox4.Checked = usekeys[3];
            checkBox5.Checked = usekeys[4];
            checkBox6.Checked = usekeys[5];
            checkBox7.Checked = usekeys[6];
            checkBox8.Checked = usekeys[7];
            checkBox9.Checked = usekeys[8];
            checkBox10.Checked = usekeys[9];
            checkBox11.Checked = usekeys[10];
            checkBox12.Checked = usekeys[11];
            return ShowDialog();
        }

        public bool[] GetValues()
        {
            return new bool[] { checkBox1.Checked, checkBox2.Checked, checkBox3.Checked, checkBox4.Checked, checkBox5.Checked, checkBox6.Checked, checkBox7.Checked, checkBox8.Checked, checkBox9.Checked, checkBox10.Checked, checkBox11.Checked, checkBox12.Checked };
        }
    }
}
