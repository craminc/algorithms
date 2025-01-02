package method;

public class CenterSpread {

    /**
     *
     */
    public static int findMaxPalindrome(String s) {
        char[] c = s.toCharArray();
        int n = c.length;
        int max = 0;
        // core
        for (int i = 0; i < 2 * n - 1; i++) {
            // simplify the judge for "a" "aa"
            int l = i / 2, r = l + i % 2;
            while (l >= 0 && r < n && c[l] == c[r]) {
                l--; r++;
            }
            max = Math.max(max, r - l - 1);
        }
        return max;
    }

    public static void main(String[] args) {
        System.out.println(findMaxPalindrome("babad"));
    }
}
