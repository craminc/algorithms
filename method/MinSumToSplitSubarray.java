package method;

public class MinSumToSplitSubarray {

    /**
     * split nums to m part, find the min of the max sum
     * for example:
     * split [1,2,3,4,5] to 2 part
     * min of the max sum is 9, split to [1,2,3,4] [5]
     */
    public static int split(int[] nums, int p) {
        int l = 0, r = 0;
        // traverse arr to record border
        for (int num : nums) {
            l = Math.max(l, num);
            r += num;
        }
        // use bisection to predict
        while (l < r) {
            // round down, so l border need to add 1 to avoid endless loop
            int m = l + r >> 1;
            // prediction is bigger, r border keep
            if (check(nums, p, m)) r = m;
            // prediction is smaller, l border add 1
            else l = m + 1;
        }
        // find the min border the can split arr to p part
        return l;
    }

    /**
     * try to split nums to as much as p part and the max sum is m
     */
    private static boolean check(int[] nums, int p, int m) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
            // split a subarray
            if (sum > m) {
                sum = num;
                p--;
            }
        }
        // p <= 0 means num left, so the split is feasible
        return p > 0;
    }

    public static void main(String[] args) {
        System.out.println(split(new int[]{7,2,5,10,8}, 2));
    }
}
