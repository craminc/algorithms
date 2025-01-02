package method;

import util.BuildArray;

import java.util.Arrays;

/**
 * 将修改保存为差分数组
 * 对差分数组求前缀和即可得到数组整体变化
 */
public class DifferArray {

    public static int[] array(int n, int[][] intervals) {
        int[] dif = new int[n + 1];

        for (int[] interval : intervals) {
            int l = interval[0];
            int r = interval[1];
            dif[l]++;
            dif[r + 1]--;
        }

        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            dif[i + 1] += dif[i];
            res[i] = dif[i];
        }

        return res;
    }

    /**
     * <a href="https://pic.leetcode-cn.com/1641658840-YrICJa-image.png">解释</a>
     * <p>
     * <a href="https://leetcode.cn/problems/increment-submatrices-by-one/description/">leetcode</a>
     */
    public static int[][] array2(int n, int[][] areas) {
        int[][] dif = new int[n + 2][n + 2];

        for (int[] area : areas) {
            int x1 = area[0];
            int y1 = area[1];
            int x2 = area[2];
            int y2 = area[3];
            dif[x1 + 1][y1 + 1]++;
            dif[x1 + 1][y2 + 2]--;
            dif[x2 + 2][y1 + 1]--;
            dif[x2 + 2][y2 + 2]++;
        }

        int[][] res = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dif[i + 1][j + 1] += dif[i][j + 1] + dif[i + 1][j] - dif[i][j];
                res[i][j] = dif[i + 1][j + 1];
            }
        }

        return res;
    }

    public static void main(String[] args) {
        System.out.println(Arrays.toString(array(5, new int[][]{
                new int[]{0, 3},
                new int[]{2, 4},
                new int[]{1, 1},
                new int[]{2, 3}
        })));

        System.out.println(Arrays.deepToString(array2(3, BuildArray.getDArray("[[1,1,2,2],[0,0,1,1]]"))));
    }
}
