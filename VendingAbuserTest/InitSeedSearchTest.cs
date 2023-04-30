using VendingAbuser;
using System.Collections.Generic;
using static VendingAbuser.Misc;


namespace Test
{
    [TestClass]
    public class InitSeedSearchTest
    {
        [TestMethod]
        public void DatecodeToYYMMDDTest()
        {
            var testcase = 0x01100423U; //2023/04/10(Mon)
            var actual = DatecodeToYYMMDD(testcase);
            var expected = new[] { 23U, 4U, 10U };
            //Console.WriteLine(String.Join(",", actual));
            CollectionAssert.AreEqual(expected, actual);

        }
        [TestMethod]
        public void TimecodeToHHMMSSTest1()
        {
            var testcase = 0x00452301U; //01:23:45(24h)(note:after12:00:00 then hh += 0x40
            var actual = TimecodeToHHMMSS(testcase);
            var expected = new[] { 01U, 23U, 45U };
            //Console.WriteLine(String.Join(",", actual));
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void TimecodeToHHMMSSTest2()
        {
            var testcase = 0x00014523U; //23:45:01(24h)(note:after12:00:00 then hh += 0x40
            var actual = TimecodeToHHMMSS(testcase);
            var expected = new[] { 23U, 45U, 01U };
            //Console.WriteLine(String.Join(",", actual));
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void YYMMDDToDatecodeTest()
        {
            var expected = new[] {23U, 4U, 13U};
            var actual = DatecodeToYYMMDD(YYMMDDToDatecode(expected));
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void HHMMSSToTimecodeTest()
        {
            var expected = new[] { 21U, 50U, 12U };
            var actual = TimecodeToHHMMSS(HHMMSSToTimecode(expected));
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GenerateDatecodeListTest()
        {
            var testcase = YYMMDDToDatecode(new[] { 23U, 4U, 13U });
            //var testcase = YYMMDDToDatecode(new[] { 0U, 1U, 1U });
            Console.WriteLine(testcase);
            Console.WriteLine(String.Join(" ", DatecodeToYYMMDD(testcase)));
            var datecodeset = new HashSet<UInt32>(GenerateDatecodeList());
            Console.WriteLine(GenerateDatecodeList()[8503]);
            Assert.IsTrue(datecodeset.Contains(testcase));
        }

        //[TestMethod]
    }
}
