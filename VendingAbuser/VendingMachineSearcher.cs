using System.Numerics;
using CUDASearch;

namespace VendingAbuser
{
    public class VendingMachineSearcher
    {
        private Vector<UInt64> mask;
        private int range;
        private int simdlength = Vector<UInt64>.Count;
        public bool useGPU;
        private VMSearcherGPU gpu = null;

        public VendingMachineSearcher(Setting cfg)
        {
            int n = cfg.N;
            int range = cfg.end - cfg.begin;
            bool useGPU = cfg.useGPU;

            this.mask = new Vector<UInt64>(Enumerable.Repeat<UInt64>((1UL << (5 * n)) - 1, simdlength).ToArray());
            this.range = range + n;
            this.useGPU = useGPU;
            if (useGPU) gpu = new VMSearcherGPU(n, range);
        }
        public VendingMachineSearcher(int n, int range, bool useGPU)
        {
            this.mask = new Vector<UInt64>(Enumerable.Repeat<UInt64>((1UL << (5 * n)) - 1, simdlength).ToArray());
            this.range = range + n;
            this.useGPU = useGPU;
            if (useGPU) gpu = new VMSearcherGPU(n, range);
        }

        public void calculate(UInt64[] seeds)
        {
            //GPUを使う場合は内部インスタンスに委譲
            if (this.useGPU)
            {
                gpu.calculate(seeds);
                return;
            }
            var add_ = new Vector<UInt64>(Enumerable.Repeat<UInt64>(0x269ec3UL, simdlength).ToArray());
            const UInt64 mul_ = 0x5d588b656c078965UL;
            var checkmask = new Vector<UInt64>(Enumerable.Repeat<UInt64>(0x1FUL, simdlength).ToArray());


            var seq = new Vector<UInt64>(new[]{1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL});

            //for (int i = 0; i < seeds.Length; i+=simdlength)
            Parallel.ForEach(Enumerable.Range(0, seeds.Length).Where(i => i % simdlength == 0), i =>
            {
                var checkcode = new Vector<UInt64>(Enumerable.Repeat<UInt64>(0xFFFFFFFFFFFFFFFFUL, simdlength).ToArray());
                var result = new Vector<UInt64>(Enumerable.Repeat<UInt64>(0x0UL, simdlength).ToArray());
                var state = new Vector<UInt64>(seeds[i..(i + simdlength)]);
                for (int j = 0; j < range; j++)
                {
                    checkcode = Vector.ShiftLeft(checkcode, 5);
                    checkcode = checkcode ^ Vector.ShiftRightLogical(state, 32+27) & checkmask;
                    result = result | Vector.Equals(checkcode & mask, Vector<UInt64>.Zero);
                    state = state * mul_;
                    state = Vector.Add(state, add_);
                }
                result = Vector.Min(result, Vector<UInt64>.One) * (Vector<UInt64>.One * (UInt64)i + seq);
                result.CopyTo(seeds,i);
            });
            return;
        }
    }
}
