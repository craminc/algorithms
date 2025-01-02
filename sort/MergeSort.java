package sort;

import java.util.Arrays;

public class MergeSort {

    private static int res = 0;

    public static void mergeSort(int[] nums) {
        merge(nums, 0, nums.length - 1);
    }

    public static void merge(int[] nums, int l, int r) {
        if (l == r) return;
        int m = l + r >> 1;
        merge(nums, l, m);
        merge(nums, m + 1, r);
        int p1 = l, p2 = m + 1;
        int[] tmp = new int[r - l + 1];
        int idx = 0;
        while (p1 <= m || p2 <= r) {
            if (p2 > r || (p1 <= m && nums[p1] <= nums[p2]))
                tmp[idx++] = nums[p1++];
            else
                tmp[idx++] = nums[p2++];
        }
        for (int i = 0; i < tmp.length; i++)
            nums[l + i] = tmp[i];
    }

    public static void main(String[] args) {
        int[] nums = new int[]{7,5,6,4,2,3,8,1};
        System.out.println(Arrays.toString(nums));
        mergeSort(nums);
        System.out.println(Arrays.toString(nums));
        System.out.println(res);
    }
}
