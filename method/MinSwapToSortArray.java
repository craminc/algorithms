package method;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MinSwapToSortArray {

    /**
     * find the loop of arr
     * min swap is arr.len - cnt(loop)
     * use the val as the next idx, finally can find a loop
     * for example:
     *  [2,1,0,4,3]
     * can find two loop, [2,1,0] [4,3]
     * each loop need len(loop.len - 1) step to sort
     * and no need to sort between each loop
     * so total need arr.len - cnt(loop) to sort whole arr
     */
    public static int minSwap(int[] arr) {
        int n = arr.length;
        int[] tmp = Arrays.copyOf(arr, n);
        // discretize
        Arrays.sort(tmp);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) map.put(tmp[i], i);
        // record if visit
        int[] vis = new int[n];
        int res = n;
        // find loop
        for (int i = 0; i < n; i++) {
            if (vis[i] != 1) res--;
            while (vis[i] != 1) {
                vis[i]++;
                // i finally back to the start
                i = map.get(arr[i]);
            }
        }
        return res;
    }

    public static void main(String[] args) {
        System.out.println(minSwap(new int[]{2,0,1,4,3}));
        System.out.println(minSwap(new int[]{7,6,8,5}));
    }
}
