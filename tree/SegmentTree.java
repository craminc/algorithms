package tree;

import util.PrintTree;

import java.util.Arrays;

public class SegmentTree {

    // origin array
    private final int[] nums;

    // nodes is a tree node array, the child of nodes[idx-1] is nodes[2*idx-1] and nodes[2*idx]
    // root node at nodes[0], idx used to position the children nodes
    private final int[] nodes;

    // lazy tag used to mark the node need to update
    private final int[] lazy;


    public SegmentTree(int[] nums) {
        this.nums = nums;
        // why need 4 multiple size array
        this.nodes = new int[nums.length * 4];
        this.lazy = new int[nums.length * 4];
        // root node idx
        build(nums, 1, 0, nums.length - 1);
    }

    private void build(int[] nums, int idx, int left, int right) {
        if (left == right) {
            // section have only one node
            nodes[idx - 1] = nums[left];
            return;
        }
        int mid = left + right >> 1;
        // left subtree
        build(nums, idx * 2, left, mid);
        // right subtree
        build(nums, idx * 2 + 1, mid + 1, right);
        // set current node val
        pushUp(idx);
    }

    /**
     * set the max value of the interval
     */
    private void pushUp(int idx) {
        nodes[idx - 1] = Math.max(nodes[idx * 2 - 1], nodes[idx * 2]);
    }

    private void pushDown(int idx) {
        if (lazy[idx - 1] != 0) {
            // update left child lazy
            lazy[idx * 2 - 1] += lazy[idx - 1];
            // update right child lazy
            lazy[idx * 2] += lazy[idx - 1];
            // update max of left child
            nodes[idx * 2 - 1] += lazy[idx - 1];
            // update max of right child
            nodes[idx * 2] += lazy[idx - 1];
            // clear lazy flag
            lazy[idx - 1] = 0;
        }
    }

    private void update(int i, int idx, int dif, int left, int right) {
        if (left == right) {
            nodes[idx - 1] += dif;
            nums[i] += dif;
            return;
        }
        int mid = left + right >> 1;
        if (i > mid) {
            update(i, idx * 2 + 1,  dif, mid + 1, right);
        } else {
            update(i, idx * 2, dif, left, mid);
        }
        pushUp(idx);
    }

    private void update(int i, int j, int idx, int dif, int left, int right) {
        if (i <= left && j >= right) {
            nodes[idx - 1] += dif;
            lazy[idx - 1] += dif;
            return;
        }
        pushDown(idx);
        int mid = left + right >> 1;
        if (i <= mid) {
            update(i, j, idx * 2,  dif, left, mid);
        }
        if (j > mid) {
            update(i, j, idx * 2 + 1, dif, mid + 1, right);
        }
        pushUp(idx);
    }

    /**
     * update nums[i] as val and update tree node
     */
    public void updateOne(int i, int val) {
        int dif = val - this.nums[i];
        this.nums[i] = val;
        update(i, 1, dif, 0, this.nums.length - 1);
    }

    /**
     * add dif to nums[i, j] and update tree node lazily
     */
    public void updateInterval(int i, int j, int dif) {
        for (int idx = i; idx <= j && idx < nums.length; idx++)
            nums[idx] += dif;
        update(i, j, 1, dif, 0, this.nums.length - 1);
    }

    /**
     * query the max value of interval nums[i, j]
     */
    public int queryInterval(int i, int j) {
        return query(i, j, 1, 0, this.nums.length - 1);
    }

    private int query(int i, int j, int idx, int left, int right) {
        if (i <= left  && j >= right) return nodes[idx - 1];
        int mid = left + right >> 1;
        int max = -1;
        pushDown(idx);
        if (i <= mid) {
            // find in left subtree
            max = Math.max(max, query(i, j, idx * 2, left, mid));
        }
        if (j > mid) {
            // find in right subtree
            max = Math.max(max, query(i, j, idx * 2 + 1, mid + 1, right));
        }
        return max;
    }

    public static void main(String[] args) {
        // init
        SegmentTree st = new SegmentTree(new int[]{1, 2, 3, 4,5,6,7,8,9,10});
        System.out.println(Arrays.toString(st.nodes));
        PrintTree.print(Arrays.stream(st.nodes).boxed().toArray());

        // update one
        st.updateOne(1, 10);
        System.out.println(Arrays.toString(st.nodes));
        PrintTree.print(Arrays.stream(st.nodes).boxed().toArray());

        // query max of interval
        System.out.println(st.queryInterval(4, 4));

        // update interval
        st.updateInterval(0, 5, 3);
        PrintTree.print(Arrays.stream(st.nodes).boxed().toArray());

        System.out.println(st.queryInterval(0, 0));
        PrintTree.print(Arrays.stream(st.nodes).boxed().toArray());
    }
}
