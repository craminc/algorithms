public class Draft {

    static class NumArray {

        private int[] bit;

        public NumArray(int[] nums) {
            bit = new int[nums.length + 1];
            for (int i = 0; i < nums.length; i++) {
                add(i, nums[i]);
            }
        }

        public void add(int index, int val) {
            for (int i = index + 1; i < bit.length; i += lowbit(i)) {
                bit[i] += val;
            }
        }

        public void update(int index, int val) {
            add(index, val - sumRange(index, index));
        }

        public int sumRange(int left, int right) {
            return sum(right + 1) - sum(left);
        }

        public int sum(int end) {
            int sum = 0;
            for (int i = end; i > 0; i -= lowbit(i)) {
                sum += bit[i];
            }
            return sum;
        }

        public int lowbit(int x) {
            return x & -x;
        }

        public static void main(String[] args) {

            NumArray numArray = new NumArray(new int[]{1, 3, 5});
            System.out.println(numArray.sumRange(0, 2));
        }
    }
}
