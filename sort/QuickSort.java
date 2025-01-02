package sort;

public class QuickSort {

    public static void quickSort(int[] nums, int l, int r) {
        if (l >= r) return;
        int start = l;
        int end = r;
        int pv = nums[l];
        while (start < end) {
            while (start < end && nums[end] <= pv) end--;
            while (start < end && nums[start] >= pv) start++;
            int tmp = nums[start];
            nums[start] = nums[end];
            nums[end] = tmp;
        }
        nums[l] = nums[start];
        nums[start] = pv;

        quickSort(nums, l, start - 1);
        quickSort(nums, start + 1, r);
    }
}
