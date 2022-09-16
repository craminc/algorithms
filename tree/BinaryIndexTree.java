package tree;

/**
 * BIT use to update an interval and find the minimum value of an interval with o(logn) time complexity
 */
public class BinaryIndexTree {

    // BIT array (bit[i] = sum(nums[i - lowbit(i)], ... , nums[i - 1])
    public int[] bit;

    public BinaryIndexTree(int[] nums) {
        // construct bit array use nums
        bit = new int[nums.length + 1];
        for (int i = 1; i <= nums.length; i++) {
            add(i, nums[i - 1]);
        }
    }

    /**
     * add val at the ith val of nums, and change bit
     */
    public void add(int idx, int val) {
        for (int i = idx; i < bit.length; i += lowbit(i)) {
            bit[i] += val;
        }
    }

    public void update(int idx, int val) {
        add(idx, (val - find(idx, idx)));
    }

    /**
     * calculate the sum of the first n terms
     */
    public int query(int term) {
        int sum = 0;
        for (int i = term; i > 0; i -= lowbit(i)) {
            sum += bit[i];
        }
        return sum;
    }

    /**
     * calculate the sum of the interval [left, right]
     */
    public int find(int left, int right) {
        return query(right) - query(left - 1);
    }

    /**
     * find the lowest bit 1 of x
     */
    private int lowbit(int x) {
        // why can do this ?
        // for example: assume binary x = 0/101100, -x = 1/101100
        // store in computer x = 0/101100, -x = (1/010011 + 1) = 1/010100
        // x & -x = 0/000100
        return x & -x;
    }

    public static void main(String[] args) {
        BinaryIndexTree bit = new BinaryIndexTree(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        System.out.println(bit.find(1, 9));
        bit.update(4, 10);
        System.out.println(bit.find(1, 9));
    }
}
