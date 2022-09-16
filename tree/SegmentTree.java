package tree;

import java.util.Arrays;

public class SegmentTree {

    // nodes is a tree node array, the child of node[i] is node[2*i] and node[2*i+1]
    // root node at nodes[1]
    private final int[] nodes;

    public SegmentTree(int[] nums) {
        // why need 4 multiple size array
        this.nodes = new int[nums.length * 4];
        build(nums, 1, 0, nums.length - 1);
    }

    private void build(int[] nums, int idx, int left, int right) {
        if (left == right) {
            nodes[idx - 1] = nums[left];
            return;
        }
        int mid = left + right >> 1;
        build(nums, idx * 2, left, mid);
        build(nums, idx * 2 + 1, mid + 1, right);
        pushUp(idx);
    }

    /**
     * set the max value of the interval
     */
    private void pushUp(int idx) {
        nodes[idx - 1] = Math.max(nodes[idx * 2 - 1], nodes[idx * 2]);
    }

    private void pushDown() {}

    /**
     * update the i of nums
     */
    private void update(int i, int val, int idx, int left, int right) {

    }

    private void query() {}

    public static void main(String[] args) {
        SegmentTree st = new SegmentTree(new int[]{1, 8, 6, 4, 3, 5});
        System.out.println(Arrays.toString(st.nodes));
    }
}
