package math;

import java.math.BigInteger;

public class Calc {

    /**
     * division algorithm
     * for example:
     *  gcd(12, 32)
     *  12 % 32 =  12
     *  32 % 12 =   8
     *  12 %  8 =   4
     *   8 %  4 =   0
     *   4 %  0 =>  b == 0
     *  return a = 4
     */
    public static int gcd(int a, int b) {
        return b == 0 ? a : gcd (b, a % b);
    }

    /**
     *  lcm(12, 32) = 12 * 32 / gcd(12, 32)
     *  = 384 / 4 = 96
     */
    public static int lcm(int a, int b) {
        return a * b / gcd(a, b);
    }

    /**
     * find the low bit 1 of x
     * for example:
     *  x = 134
     *  x => 128 + 6 => 10000110
     * -x => 01111001 + 1 = 01111010
     *  x & -x = 00000010 = 2
     */
    public static int lowbit(int x) {
        return x & -x;
    }

    /**
     * calculate a ^ b with mod (b was too large)
     */
    public static int quickPow(int a, int b, int mod) {
        long ans = 1;
        long x = a;
        while (b > 0) {
            if ((b & 1) == 1) ans = ans * x % mod;
            x = x * x % mod;
            b >>= 1;
        }
        return (int) ans;
    }

    /**
     * calculate Gray Code for n
     * Gray Code every pair of adjacent integers differs by exactly one bit
     * for example 00 01 11 10
     *      (n)2 -> xxxx01111 ->            (n ^ (n >> 1))2 -> xxxx(0/1)1000
     *  (n + 1)2 -> xxxx10000 ->  (n + 1 ^ ((n + 1) >> 1))2 -> xxxx(1/0)1000
     */
    public static int grayCode(int n) {
        return n ^ (n >> 1);
    }

    public static void main(String[] args) {
        int a = 12, b = 32;
        System.out.printf("gcd(%d, %d) = %d\n", a, b, gcd(a, b));
        System.out.printf("lcm(%d, %d) = %d\n", a, b, lcm(a, b));
        System.out.println(lowbit(134));
        System.out.println(quickPow(2, 12412415, (int) 1e9 + 7));
        System.out.println(BigInteger.valueOf(2).modPow(BigInteger.valueOf(12412415), BigInteger.valueOf((int) 1e9 + 7)));
        System.out.println(grayCode(15));
    }
}
