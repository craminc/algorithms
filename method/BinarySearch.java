package method;

/**
 * @Author: cramin
 * @Date: 2024/1/26 17:53
 * @Desc:
 */
public class BinarySearch {

    /**
     * 二分查询数组中第一个小于 val 的元素
     */
    public static int findL(int[] arr, int val) {
        int l = 0, r = arr.length - 1;
        while (l < r) {
            // 向 r 侧移动
            int m = l + r + 1 >> 1;
            if (arr[m] <= val) l = m;
            // r 减少
            else r = m - 1;
        }
        return l;
    }

    /**
     * 二分查询数组中第一个大于 val 的元素
     */
    public static int findR(int[] arr, int val) {
        int l = 0, r = arr.length - 1;
        while (l < r) {
            // 向 l 侧移动
            int m = l + r >> 1;
            if (arr[m] > val) r = m;
            // l 增加
            else l = m + 1;
        }
        return l;
    }

    public static void main(String[] args) {
        int[] arr = {1, 2, 4, 4, 4, 6, 7};
        int l = findL(arr, 8);
        int r = findR(arr, 4);
        System.out.println(l);
        System.out.println(r);
    }
}
