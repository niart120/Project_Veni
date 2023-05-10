namespace VendingAbuser
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            DataGridViewCellStyle dataGridViewCellStyle1 = new DataGridViewCellStyle();
            DataGridViewCellStyle dataGridViewCellStyle2 = new DataGridViewCellStyle();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            groupBox1 = new GroupBox();
            comboBox2 = new ComboBox();
            groupBox2 = new GroupBox();
            comboBox1 = new ComboBox();
            groupBox3 = new GroupBox();
            numericUpDown7 = new NumericUpDown();
            numericUpDown6 = new NumericUpDown();
            numericUpDown5 = new NumericUpDown();
            numericUpDown4 = new NumericUpDown();
            numericUpDown3 = new NumericUpDown();
            numericUpDown2 = new NumericUpDown();
            groupBox4 = new GroupBox();
            label1 = new Label();
            numericUpDown1 = new NumericUpDown();
            progressBar1 = new ProgressBar();
            button2 = new Button();
            button3 = new Button();
            checkBox1 = new CheckBox();
            groupBox5 = new GroupBox();
            button1 = new Button();
            checkBox2 = new CheckBox();
            groupBox6 = new GroupBox();
            label2 = new Label();
            numericUpDown9 = new NumericUpDown();
            numericUpDown8 = new NumericUpDown();
            dataGridView1 = new DataGridView();
            Column1 = new DataGridViewTextBoxColumn();
            Column2 = new DataGridViewTextBoxColumn();
            Column3 = new DataGridViewTextBoxColumn();
            Column4 = new DataGridViewTextBoxColumn();
            Column5 = new DataGridViewTextBoxColumn();
            Column6 = new DataGridViewTextBoxColumn();
            groupBox7 = new GroupBox();
            label3 = new Label();
            dateTimePicker2 = new DateTimePicker();
            dateTimePicker1 = new DateTimePicker();
            groupBox1.SuspendLayout();
            groupBox2.SuspendLayout();
            groupBox3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)numericUpDown7).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown6).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown5).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown4).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown3).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown2).BeginInit();
            groupBox4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)numericUpDown1).BeginInit();
            groupBox5.SuspendLayout();
            groupBox6.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)numericUpDown9).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown8).BeginInit();
            ((System.ComponentModel.ISupportInitialize)dataGridView1).BeginInit();
            groupBox7.SuspendLayout();
            SuspendLayout();
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(comboBox2);
            groupBox1.Location = new Point(165, 16);
            groupBox1.Name = "groupBox1";
            groupBox1.Size = new Size(170, 94);
            groupBox1.TabIndex = 1;
            groupBox1.TabStop = false;
            groupBox1.Text = "ROM";
            // 
            // comboBox2
            // 
            comboBox2.DropDownStyle = ComboBoxStyle.DropDownList;
            comboBox2.FormattingEnabled = true;
            comboBox2.ImeMode = ImeMode.Disable;
            comboBox2.Items.AddRange(new object[] { "ブラック", "ホワイト", "ブラック2", "ホワイト2" });
            comboBox2.Location = new Point(19, 38);
            comboBox2.Name = "comboBox2";
            comboBox2.Size = new Size(130, 40);
            comboBox2.TabIndex = 0;
            // 
            // groupBox2
            // 
            groupBox2.Controls.Add(comboBox1);
            groupBox2.Location = new Point(12, 16);
            groupBox2.Name = "groupBox2";
            groupBox2.Size = new Size(148, 94);
            groupBox2.TabIndex = 0;
            groupBox2.TabStop = false;
            groupBox2.Text = "Hardware";
            // 
            // comboBox1
            // 
            comboBox1.DropDownStyle = ComboBoxStyle.DropDownList;
            comboBox1.FormattingEnabled = true;
            comboBox1.ImeMode = ImeMode.Disable;
            comboBox1.Items.AddRange(new object[] { "DS", "DSLite", "DSi", "3DS" });
            comboBox1.Location = new Point(18, 34);
            comboBox1.Name = "comboBox1";
            comboBox1.Size = new Size(112, 40);
            comboBox1.TabIndex = 0;
            // 
            // groupBox3
            // 
            groupBox3.Controls.Add(numericUpDown7);
            groupBox3.Controls.Add(numericUpDown6);
            groupBox3.Controls.Add(numericUpDown5);
            groupBox3.Controls.Add(numericUpDown4);
            groupBox3.Controls.Add(numericUpDown3);
            groupBox3.Controls.Add(numericUpDown2);
            groupBox3.Location = new Point(12, 112);
            groupBox3.Name = "groupBox3";
            groupBox3.Size = new Size(550, 84);
            groupBox3.TabIndex = 4;
            groupBox3.TabStop = false;
            groupBox3.Text = "MAC Address";
            // 
            // numericUpDown7
            // 
            numericUpDown7.Hexadecimal = true;
            numericUpDown7.ImeMode = ImeMode.Off;
            numericUpDown7.Location = new Point(468, 37);
            numericUpDown7.Maximum = new decimal(new int[] { 255, 0, 0, 0 });
            numericUpDown7.Name = "numericUpDown7";
            numericUpDown7.Size = new Size(68, 39);
            numericUpDown7.TabIndex = 5;
            numericUpDown7.TextAlign = HorizontalAlignment.Right;
            numericUpDown7.Enter += numericUpDown7_Enter;
            // 
            // numericUpDown6
            // 
            numericUpDown6.Hexadecimal = true;
            numericUpDown6.ImeMode = ImeMode.Off;
            numericUpDown6.Location = new Point(378, 37);
            numericUpDown6.Maximum = new decimal(new int[] { 255, 0, 0, 0 });
            numericUpDown6.Name = "numericUpDown6";
            numericUpDown6.Size = new Size(68, 39);
            numericUpDown6.TabIndex = 4;
            numericUpDown6.TextAlign = HorizontalAlignment.Right;
            numericUpDown6.Enter += numericUpDown6_Enter;
            // 
            // numericUpDown5
            // 
            numericUpDown5.Hexadecimal = true;
            numericUpDown5.ImeMode = ImeMode.Off;
            numericUpDown5.Location = new Point(288, 37);
            numericUpDown5.Maximum = new decimal(new int[] { 255, 0, 0, 0 });
            numericUpDown5.Name = "numericUpDown5";
            numericUpDown5.Size = new Size(68, 39);
            numericUpDown5.TabIndex = 3;
            numericUpDown5.TextAlign = HorizontalAlignment.Right;
            numericUpDown5.Enter += numericUpDown5_Enter;
            // 
            // numericUpDown4
            // 
            numericUpDown4.Hexadecimal = true;
            numericUpDown4.ImeMode = ImeMode.Off;
            numericUpDown4.Location = new Point(198, 37);
            numericUpDown4.Maximum = new decimal(new int[] { 255, 0, 0, 0 });
            numericUpDown4.Name = "numericUpDown4";
            numericUpDown4.Size = new Size(68, 39);
            numericUpDown4.TabIndex = 2;
            numericUpDown4.TextAlign = HorizontalAlignment.Right;
            numericUpDown4.Enter += numericUpDown4_Enter;
            // 
            // numericUpDown3
            // 
            numericUpDown3.Hexadecimal = true;
            numericUpDown3.ImeMode = ImeMode.Off;
            numericUpDown3.Location = new Point(108, 37);
            numericUpDown3.Maximum = new decimal(new int[] { 255, 0, 0, 0 });
            numericUpDown3.Name = "numericUpDown3";
            numericUpDown3.Size = new Size(68, 39);
            numericUpDown3.TabIndex = 1;
            numericUpDown3.TextAlign = HorizontalAlignment.Right;
            numericUpDown3.Enter += numericUpDown3_Enter;
            // 
            // numericUpDown2
            // 
            numericUpDown2.Hexadecimal = true;
            numericUpDown2.ImeMode = ImeMode.Off;
            numericUpDown2.Location = new Point(18, 37);
            numericUpDown2.Maximum = new decimal(new int[] { 255, 0, 0, 0 });
            numericUpDown2.Name = "numericUpDown2";
            numericUpDown2.Size = new Size(68, 39);
            numericUpDown2.TabIndex = 0;
            numericUpDown2.TextAlign = HorizontalAlignment.Right;
            numericUpDown2.Enter += numericUpDown2_Enter;
            // 
            // groupBox4
            // 
            groupBox4.Controls.Add(label1);
            groupBox4.Controls.Add(numericUpDown1);
            groupBox4.Location = new Point(340, 16);
            groupBox4.Name = "groupBox4";
            groupBox4.Size = new Size(82, 94);
            groupBox4.TabIndex = 2;
            groupBox4.TabStop = false;
            groupBox4.Text = "N=";
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(72, 41);
            label1.Name = "label1";
            label1.Size = new Size(0, 32);
            label1.TabIndex = 1;
            // 
            // numericUpDown1
            // 
            numericUpDown1.Location = new Point(6, 39);
            numericUpDown1.Maximum = new decimal(new int[] { 13, 0, 0, 0 });
            numericUpDown1.Minimum = new decimal(new int[] { 5, 0, 0, 0 });
            numericUpDown1.Name = "numericUpDown1";
            numericUpDown1.Size = new Size(64, 39);
            numericUpDown1.TabIndex = 0;
            numericUpDown1.TextAlign = HorizontalAlignment.Right;
            numericUpDown1.Value = new decimal(new int[] { 6, 0, 0, 0 });
            // 
            // progressBar1
            // 
            progressBar1.Location = new Point(12, 410);
            progressBar1.Name = "progressBar1";
            progressBar1.Size = new Size(550, 46);
            progressBar1.TabIndex = 5;
            // 
            // button2
            // 
            button2.Location = new Point(26, 466);
            button2.Name = "button2";
            button2.Size = new Size(418, 46);
            button2.TabIndex = 7;
            button2.Text = "Search";
            button2.UseVisualStyleBackColor = true;
            button2.Click += button2_Click;
            // 
            // button3
            // 
            button3.Enabled = false;
            button3.Location = new Point(454, 466);
            button3.Name = "button3";
            button3.Size = new Size(100, 46);
            button3.TabIndex = 8;
            button3.Text = "Cancel";
            button3.UseVisualStyleBackColor = true;
            button3.Click += button3_Click;
            // 
            // checkBox1
            // 
            checkBox1.AutoSize = true;
            checkBox1.BackColor = Color.Transparent;
            checkBox1.Location = new Point(428, 56);
            checkBox1.Name = "checkBox1";
            checkBox1.Size = new Size(146, 36);
            checkBox1.TabIndex = 3;
            checkBox1.Text = "useCUDA";
            checkBox1.UseVisualStyleBackColor = false;
            // 
            // groupBox5
            // 
            groupBox5.Controls.Add(button1);
            groupBox5.Controls.Add(checkBox2);
            groupBox5.Location = new Point(258, 202);
            groupBox5.Name = "groupBox5";
            groupBox5.Size = new Size(304, 96);
            groupBox5.TabIndex = 6;
            groupBox5.TabStop = false;
            groupBox5.Text = "Keys";
            // 
            // button1
            // 
            button1.Location = new Point(190, 36);
            button1.Name = "button1";
            button1.Size = new Size(100, 46);
            button1.TabIndex = 1;
            button1.Text = "Setting";
            button1.UseVisualStyleBackColor = true;
            button1.Click += button1_Click;
            // 
            // checkBox2
            // 
            checkBox2.AutoSize = true;
            checkBox2.Location = new Point(12, 42);
            checkBox2.Name = "checkBox2";
            checkBox2.Size = new Size(178, 36);
            checkBox2.TabIndex = 0;
            checkBox2.Text = "useKeyInput";
            checkBox2.UseVisualStyleBackColor = true;
            // 
            // groupBox6
            // 
            groupBox6.Controls.Add(label2);
            groupBox6.Controls.Add(numericUpDown9);
            groupBox6.Controls.Add(numericUpDown8);
            groupBox6.Location = new Point(12, 202);
            groupBox6.Name = "groupBox6";
            groupBox6.Size = new Size(240, 96);
            groupBox6.TabIndex = 5;
            groupBox6.TabStop = false;
            groupBox6.Text = "SearchAdv.Range";
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.BackColor = Color.Transparent;
            label2.Location = new Point(104, 46);
            label2.Name = "label2";
            label2.Size = new Size(38, 32);
            label2.TabIndex = 2;
            label2.Text = "～";
            // 
            // numericUpDown9
            // 
            numericUpDown9.Location = new Point(145, 43);
            numericUpDown9.Maximum = new decimal(new int[] { 9999, 0, 0, 0 });
            numericUpDown9.Name = "numericUpDown9";
            numericUpDown9.Size = new Size(80, 39);
            numericUpDown9.TabIndex = 1;
            numericUpDown9.TextAlign = HorizontalAlignment.Right;
            numericUpDown9.Value = new decimal(new int[] { 150, 0, 0, 0 });
            // 
            // numericUpDown8
            // 
            numericUpDown8.Location = new Point(18, 43);
            numericUpDown8.Maximum = new decimal(new int[] { 9999, 0, 0, 0 });
            numericUpDown8.Name = "numericUpDown8";
            numericUpDown8.Size = new Size(84, 39);
            numericUpDown8.TabIndex = 0;
            numericUpDown8.TextAlign = HorizontalAlignment.Right;
            numericUpDown8.Value = new decimal(new int[] { 50, 0, 0, 0 });
            // 
            // dataGridView1
            // 
            dataGridView1.AllowUserToAddRows = false;
            dataGridView1.AllowUserToDeleteRows = false;
            dataGridView1.AllowUserToResizeColumns = false;
            dataGridView1.AllowUserToResizeRows = false;
            dataGridView1.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            dataGridView1.Columns.AddRange(new DataGridViewColumn[] { Column1, Column2, Column3, Column4, Column5, Column6 });
            dataGridView1.Location = new Point(568, 2);
            dataGridView1.Name = "dataGridView1";
            dataGridView1.ReadOnly = true;
            dataGridView1.RowHeadersVisible = false;
            dataGridView1.RowHeadersWidth = 82;
            dataGridView1.RowTemplate.Height = 41;
            dataGridView1.ScrollBars = ScrollBars.Vertical;
            dataGridView1.Size = new Size(900, 518);
            dataGridView1.TabIndex = 9;
            // 
            // Column1
            // 
            Column1.HeaderText = "InitSeed";
            Column1.MinimumWidth = 10;
            Column1.Name = "Column1";
            Column1.ReadOnly = true;
            Column1.Width = 240;
            // 
            // Column2
            // 
            Column2.HeaderText = "Date";
            Column2.MinimumWidth = 10;
            Column2.Name = "Column2";
            Column2.ReadOnly = true;
            Column2.Width = 140;
            // 
            // Column3
            // 
            Column3.HeaderText = "Time";
            Column3.MinimumWidth = 10;
            Column3.Name = "Column3";
            Column3.ReadOnly = true;
            Column3.Width = 110;
            // 
            // Column4
            // 
            dataGridViewCellStyle1.Alignment = DataGridViewContentAlignment.MiddleRight;
            Column4.DefaultCellStyle = dataGridViewCellStyle1;
            Column4.HeaderText = "Timer0";
            Column4.MinimumWidth = 10;
            Column4.Name = "Column4";
            Column4.ReadOnly = true;
            Column4.Width = 92;
            // 
            // Column5
            // 
            dataGridViewCellStyle2.Alignment = DataGridViewContentAlignment.MiddleRight;
            Column5.DefaultCellStyle = dataGridViewCellStyle2;
            Column5.HeaderText = "Advance";
            Column5.MinimumWidth = 10;
            Column5.Name = "Column5";
            Column5.ReadOnly = true;
            Column5.Width = 110;
            // 
            // Column6
            // 
            Column6.HeaderText = "KeyInput";
            Column6.MinimumWidth = 10;
            Column6.Name = "Column6";
            Column6.ReadOnly = true;
            Column6.Width = 170;
            // 
            // groupBox7
            // 
            groupBox7.Controls.Add(label3);
            groupBox7.Controls.Add(dateTimePicker2);
            groupBox7.Controls.Add(dateTimePicker1);
            groupBox7.Location = new Point(12, 301);
            groupBox7.Name = "groupBox7";
            groupBox7.Size = new Size(550, 102);
            groupBox7.TabIndex = 10;
            groupBox7.TabStop = false;
            groupBox7.Text = "SearchDateRange";
            // 
            // label3
            // 
            label3.AutoSize = true;
            label3.BackColor = Color.Transparent;
            label3.Location = new Point(258, 49);
            label3.Name = "label3";
            label3.Size = new Size(38, 32);
            label3.TabIndex = 2;
            label3.Text = "～";
            // 
            // dateTimePicker2
            // 
            dateTimePicker2.Location = new Point(296, 46);
            dateTimePicker2.MaxDate = new DateTime(2099, 12, 31, 0, 0, 0, 0);
            dateTimePicker2.MinDate = new DateTime(2000, 1, 1, 0, 0, 0, 0);
            dateTimePicker2.Name = "dateTimePicker2";
            dateTimePicker2.Size = new Size(240, 39);
            dateTimePicker2.TabIndex = 0;
            // 
            // dateTimePicker1
            // 
            dateTimePicker1.Location = new Point(18, 46);
            dateTimePicker1.MaxDate = new DateTime(2099, 12, 31, 0, 0, 0, 0);
            dateTimePicker1.MinDate = new DateTime(2000, 1, 1, 0, 0, 0, 0);
            dateTimePicker1.Name = "dateTimePicker1";
            dateTimePicker1.Size = new Size(240, 39);
            dateTimePicker1.TabIndex = 0;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(13F, 32F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1474, 529);
            Controls.Add(groupBox7);
            Controls.Add(dataGridView1);
            Controls.Add(groupBox6);
            Controls.Add(groupBox5);
            Controls.Add(checkBox1);
            Controls.Add(button3);
            Controls.Add(groupBox2);
            Controls.Add(button2);
            Controls.Add(progressBar1);
            Controls.Add(groupBox3);
            Controls.Add(groupBox4);
            Controls.Add(groupBox1);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "Form1";
            Text = "VendingAbuser";
            Load += Form1_Load;
            groupBox1.ResumeLayout(false);
            groupBox2.ResumeLayout(false);
            groupBox3.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)numericUpDown7).EndInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown6).EndInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown5).EndInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown4).EndInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown3).EndInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown2).EndInit();
            groupBox4.ResumeLayout(false);
            groupBox4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)numericUpDown1).EndInit();
            groupBox5.ResumeLayout(false);
            groupBox5.PerformLayout();
            groupBox6.ResumeLayout(false);
            groupBox6.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)numericUpDown9).EndInit();
            ((System.ComponentModel.ISupportInitialize)numericUpDown8).EndInit();
            ((System.ComponentModel.ISupportInitialize)dataGridView1).EndInit();
            groupBox7.ResumeLayout(false);
            groupBox7.PerformLayout();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private GroupBox groupBox1;
        private ComboBox comboBox2;
        private GroupBox groupBox2;
        private ComboBox comboBox1;
        private GroupBox groupBox3;
        private GroupBox groupBox4;
        private Label label1;
        private NumericUpDown numericUpDown1;
        private ProgressBar progressBar1;
        private Button button1;
        private Button button2;
        private Button button3;
        private NumericUpDown numericUpDown7;
        private NumericUpDown numericUpDown6;
        private NumericUpDown numericUpDown5;
        private NumericUpDown numericUpDown4;
        private NumericUpDown numericUpDown3;
        private NumericUpDown numericUpDown2;
        private CheckBox checkBox1;
        private GroupBox groupBox5;
        private CheckBox checkBox2;
        private GroupBox groupBox6;
        private Label label2;
        private NumericUpDown numericUpDown9;
        private NumericUpDown numericUpDown8;
        private DataGridView dataGridView1;
        private DataGridViewTextBoxColumn Column1;
        private DataGridViewTextBoxColumn Column2;
        private DataGridViewTextBoxColumn Column3;
        private DataGridViewTextBoxColumn Column4;
        private DataGridViewTextBoxColumn Column5;
        private DataGridViewTextBoxColumn Column6;
        private GroupBox groupBox7;
        private DateTimePicker dateTimePicker2;
        private DateTimePicker dateTimePicker1;
        private Label label3;
    }
}