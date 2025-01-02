package util;

import java.util.Arrays;

public class BuildArray {

    public static int[] getArray(String str) {
        if (str.charAt(0) == '[' && str.charAt(str.length() - 1) == ']')
            str = str.substring(1, str.length() - 1);
        String[] split = str.split(",");

        int[] res = new int[split.length];
        for (int i = 0; i < res.length; i++)
            res[i] = Integer.parseInt(split[i]);

        return res;
    }

    public static char[] getCharArr(String str) {
        if (str.charAt(0) == '[' && str.charAt(str.length() - 1) == ']')
            str = str.substring(1, str.length() - 1);
        String[] split = str.split(",");

        char[] res = new char[split.length];
        for (int i = 0; i < res.length; i++)
            res[i] = split[i].charAt(1);

        return res;
    }

    public static int[][] getDArray(String str) {
        String substr = str.substring(2, str.length() - 2);
        String[] split = substr.split("],\\[");

        int[][] res = new int[split.length][];
        for (int i = 0; i < res.length; i++)
            res[i] = getArray(split[i]);
        return res;
    }

    public static char[][] getDCharArray(String str) {
        String substr = str.substring(2, str.length() - 2);
        String[] split = substr.split("],\\[");

        char[][] res = new char[split.length][];
        for (int i = 0; i < res.length; i++)
            res[i] = getCharArr(split[i]);
        return res;
    }

    public static void main(String[] args) {
        System.out.println(Arrays.toString(BuildArray.getArray("[1,2,3,4]")));
        System.out.println(Arrays.deepToString(BuildArray.getDArray("[[-37,-50,-3,44],[-37,46,13,-32],[47,-42,-3,-40],[-17,-22,-39,24]]")));
    }
}
