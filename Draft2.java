import util.BuildArray;

import java.util.*;

public class Draft2 {

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
}

    public static void main(String[] args) {
        Draft2 draft = new Draft2();
        draft.findSubstring("aaaaaaaaaaaaaa", new String[]{"aa","aa"});
    }

    public List<Integer> findSubstring(String s, String[] words) {
        int slen = s.length(), wlen = words[0].length(), wNum = words.length;
        Map<String, Integer> cntMap = new HashMap<>();
        for (String word : words)
            cntMap.compute(word, (k, v) -> v == null ? 1 : v + 1);

        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < wlen; i++) {
            Map<String, Integer> curMap = new HashMap<>();
            int pNum = 0, start = i;
            for (int j = i; j + (wNum - pNum) * wlen <= slen && j + wlen < slen; j += wlen) {
                String sub = s.substring(j, j + wlen);
                if (cntMap.get(sub) == null) {
                    pNum = 0;
                    curMap = new HashMap<>();
                    start = j + wlen;
                    continue;
                }
                curMap.compute(sub, (k, v) -> v == null ? 1 : v + 1);
                pNum++;
                if (curMap.get(sub) > cntMap.get(sub)) {
                    for (int k = start; k < j; k += wlen) {
                        String preSub = s.substring(k, k + wlen);
                        curMap.put(preSub, curMap.get(preSub) - 1);
                        pNum--;
                        if (preSub.equals(sub)) {
                            start = k + wlen;
                            break;
                        }
                    }
                }
                if (pNum == wNum) res.add(start);
            }
        }
        return res;
    }

    public int countSubarrays(int[] nums, int k) {
        int n = nums.length;
        int[] map = new int[2 * n];
        // 小于k -1, 等于k 0, 大于k 1
        // 转变为找到子区间和为 0 或 1 且区间包括当前元素
        int pre = n, cnt = 1;
        boolean f = false;
        for (int num : nums) {
            if (num == k) f = true;
            else pre += num > k ? 1 : -1;
            if (f) cnt += map[pre] + map[pre - 1];
            else map[pre]++;
        }

        return cnt;
    }

    public String[] findLongestSubarray(String[] array) {
        // 区间和为0的最大子区间
        int n = array.length;
        int[] sum = new int[n + 1];
        int[] map = new int[2 * n + 1];
        sum[0] = n; map[0] = n - 1;
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + (array[i].charAt(0) >> 6 & 1) * 2 - 1;
            // 记录最右区间端点
            map[sum[i + 1]] = i;
        }

        int start = 0, end = -1;
        for (int i = 0; i < n - (end - start + 1); i++) {
            if (map[sum[i]] - i > end - start) {
                start = i;
                end = map[sum[i]];
            }
        }
        String[] res = new String[end - start + 1];
        if (end + 1 - start >= 0)
            System.arraycopy(array, start, res, 0, end - start + 1);

        return res;
    }

    public int minSubarray(int[] nums, int p) {
        int n = nums.length;
        int[] pre = new int[n + 1];
        int[] map = new int[p];
        Arrays.fill(map, n + 1);
        for (int i = 0; i < n; i++) {
            pre[i + 1] = (pre[i] + nums[i]) % p;
            // the rightmost index for every prefix sum % p.
            map[pre[i]] = i;
        }
        int min = n;
        for (int i = n; i > 0; i--) {
            int m = ((pre[i] + p) - pre[n]) % p;
            if (map[m] <= i) min = Math.min(min, i - map[m]);
        }
        return min == n ? -1 : min;
    }

    public boolean verifySequenceOfBST(int[] arr) {
        if (arr.length == 0) return false;
        verify(arr, 0, arr.length - 1);
        return true;
    }

    /**
     * 验证是否能找到一个中轴位置，左边都小于根，右边都大于根
     */
    public boolean verify(int[] arr, int l, int r) {
        if (l >= r) return true;
        // 最后一个节点作为根
        int idx = r - 1;
        // 右侧部分
        while (idx >= l && arr[idx] > arr[r]) idx--;
        int mid = idx;
        // 判断左侧部分是否合法
        while (idx >= l) {
            if (arr[idx--] > arr[r]) return false;
        }
        // 分别判断左右两部分是否能构成搜索二叉树
        return verify(arr, l, mid) && verify(arr, mid + 1, r - 1);
    }

    public String printBin(double num) {
        char[] c = new char[32];
        c[0] = '0'; c[1] = '.';
        int idx = 2;
        while (num > 0 && idx < 32) {
            if ((num *= 2) >= 1) {
                c[idx++] = '1';
                num -= 1;
            } else {
                c[idx++] = '0';
            }
        }
        if (idx == 32 && num > 0) return "ERROR";
        return new String(c, 0, idx);
    }

    // 队列队尾入队，队头出队
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < k; i++) offer(queue, nums[i]);
        int[] res = new int[nums.length - k + 1];
        res[0] = queue.peekFirst();
        for (int i = 1, pre = res[0]; i < res.length; i++) {
            // 如果将要出队的是最大的元素，则队头出队
            if (nums[i - 1] == pre) queue.pollFirst();
            // 新入队一个元素
            offer(queue, nums[i + k]);
            res[i] = queue.peekFirst();
            pre = res[i];
        }
        return res;
    }

    public void offer(Deque<Integer> queue, int val) {
        // 维护一个递增的单调队列（队头元素最大）
        while (!queue.isEmpty() && val > queue.peekLast())
            // 从队尾开始判断，如果队尾元素比入队元素小，则出队
            queue.pollLast();
        queue.addLast(val);
    }

    public int movesToMakeZigzag(int[] nums) {
        // 要么缩小偶数下标 要么缩小奇数下标（因为只能减少元素不能增加元素）
        int cnt1 = 0, cnt2 = 0, n = nums.length;
        for (int i = 0; i < n; i += 2) {
            int n1 = nums[i];
            if (i - 1 >= 0 && n1 >= nums[i - 1]) {
                cnt1 += n1 - nums[i - 1] + 1;
                n1 = nums[i - 1] - 1;
            }
            if (i + 1 < n && n1 >= nums[i + 1]) {
                cnt1 += n1 - nums[i + 1] + 1;
            }
            if (i + 1 < n) {
                int n2 = nums[i + 1];
                if (n2 >= nums[i]) {
                    cnt2 += n2 - nums[i] + 1;
                    n2 = nums[i] - 1;
                }
                if (i + 2 < n && n2 >= nums[i + 2]) {
                    cnt2 += n2 - nums[i + 2] + 1;
                }
            }
        }
        return Math.min(cnt1, cnt2);
    }
    public List<Integer> circularPermutation(int n, int start) {
        int[] arr = new int[1 << n];
        arr[1] = 1;
        for (int i = 2; i < arr.length;) {
            int j = i, k = i;
            while (j > 0)
                arr[i++] = k + arr[j--];
        }

        List<Integer> res = new ArrayList<>();
        for (int a : arr) res.add(a);

        return res;
    }

    public int minTaps(int n, int[] ranges) {
        int len = ranges.length;
        int[][] gaps = new int[len][2];
        for (int i = 0; i < len; i++) {
            int range = ranges[i];
            gaps[i] = new int[]{i - range, i + range};
        }

        // Arrays.sort(gaps, (a, b) -> a[0] - b[0]);

        int l = 0, cnt = 0, i = 0;
        while (i < len) {
            // 找到包含某个区间，且右端点最远的gap
            int r = l;
            cnt++;
            while (i < len && gaps[i][0] <= l) {
                r = Math.max(r, gaps[i++][1]);
                if (r >= n) return cnt;
            }
            if (l == r) return -1;
            l = r;
        }
        return -1;
    }
    private final static int[] prime = new int[]{2,3,5,7,11,13,17,19,23,29};
    private final static int mod = (int) 1e9 + 7, max = 31, len = 10;
    private final static int[] mask = new int[max];
    static {
        for (int i = 2; i < max; i++) {
            for (int j = 0; j < len; j++) {
                int p = prime[j];
                if (i % p == 0) {
                    if (i % (p * p) == 0) { mask[i] = -1; break; }
                    mask[i] |= 1 << j;
                }
            }
        }
    }
    public int numberOfGoodSubsets(int[] nums) {
        int[] cnt = new int[max];
        for (int num : nums) cnt[num]++;
        int mms = 1 << len;
        int[] dp = new int[mms];
        dp[0] = 1;
        for (int i = 1; i < max; i++) {
            int ms = mask[i];
            if (ms < 0 || cnt[i] == 0) continue;
            for (int n = mms - 1; n >= ms; n--) {
                if ((n | ms) == n)
                    dp[n] = (int) (dp[n] + (long) dp[n ^ ms] * cnt[i]) % mod;
            }
        }
        long res = 0;
        for (int i = 1; i < mms; i++) res = (res + dp[i]) % mod;

        return (int) res;
    }

    public int longestWPI(int[] hours) {
        int n = hours.length;
        int[] pre = new int[n + 1];
        for (int i = 0; i < n; i++)
            pre[i + 1] = pre[i] + (hours[i] > 8 ? 1 : -1);
        // 单调递减栈（记录前缀和的下标 初始栈顶为0）
        int[] stack = new int[n + 1];
        int sp = 0;
        // 记录前面有多长的良好区间
        int[] map = new int[n + 1];
        // 找到区间和大于0的最大区间长度
        int max = 0;
        for (int i = 0; i < n; i++) {
            while (sp > -1 && pre[i + 1] > pre[stack[sp]]) {
                int len = i + 1 - stack[sp] + map[i];
                max = Math.max(max, len);
                map[i + 1] = len;
                sp--;
            }
            stack[++sp] = i + 1;
        }

        return max;
    }

    public int[][] substringXorQueries(String s, int[][] queries) {
        // preprocess the substring of s
        Map<Integer, int[]> map = new HashMap<>();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            if (ca[i] == '0') {
                map.merge(0, new int[]{i, i}, (o, n) -> o);
                continue;
            }
            int num = 0;
            for (int j = i; j - i < 31 && j < ca.length; j++) {
                num = (num << 1) + (ca[j] - '0');
                map.merge(num, new int[]{i, j}, (o, n) -> o);
            }
        }
        int[] not = new int[]{-1, -1};
        int[][] res = new int[queries.length][];
        for (int i = 0; i < queries.length; i++) {
            int q = queries[i][0] ^ queries[i][1];
            res[i] = map.getOrDefault(q, not);
        }
        return res;
    }

    public long countFairPairs(int[] nums, int lower, int upper) {
        Arrays.sort(nums);
        long cnt = 0;
        int l = nums.length - 1, r = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {
            int low = lower - nums[i];
            int high = upper - nums[i];
            l = findL(nums, i + 1, Math.max(l, i + 1), low);
            r = findR(nums, i + 1, r, high);
            if (l == i || r == i) break;
            cnt += r - l;
        }

        return cnt;
    }

    // 找到第一个大于等于n的数
    public int findL(int[] nums, int l, int r, int n) {
        r = Math.min(r, nums.length - 1);
        while (l < r) {
            int m = l + r >> 1;
            if (nums[m] >= n) r = m;
            else l = m + 1;
        }
        return nums[r] >= n ? r : r + 1;
    }

    // 找到第一个大于n的数
    public int findR(int[] nums, int l, int r, int n) {
        r = Math.min(r, nums.length - 1);
        while (l < r) {
            int m = l + r >> 1;
            if (nums[m] > n) r = m;
            else l = m + 1;
        }
        return nums[r] > n ? r : r + 1;
    }

    public int balancedString(String s) {
        int[] cnt = new int[26];
        char[] ca = s.toCharArray();
        int n = ca.length;
        int avg = n / 4;
        for (char c : ca) cnt[c - 'A']++;
        int l = 0, r = 0;
        int min = n;
        while (r < n) {
            while (r < n && cnt[ca[r] - 'A'] > avg) cnt[ca[r++] - 'A']--;
            while (l < r && cnt[ca[l] - 'A'] < avg) cnt[ca[l++] - 'A']++;
            if (balanced(cnt, avg)) min = Math.min(min, r - l);
            if (r < n) cnt[ca[r++] - 'A']--;
        }
        while (l < r && cnt[ca[l] - 'A'] < avg) cnt[ca[l++] - 'A']++;
        if (balanced(cnt, avg)) min = Math.min(min, r - l);
        return min;
    }

    public boolean balanced(int[] cnt, int avg) {
        return cnt['Q' - 'A'] <= avg && cnt['W' - 'A'] <= avg &&
                cnt['E' - 'A'] <= avg && cnt['R' - 'A'] <= avg;
    }

    public List<String> removeSubfolders(String[] folder) {
        Trie root = new Trie("");
        for (String f : folder) {
            String[] dir = f.split("/");
            Trie cur = root;
            for (int i = 1; i < dir.length; i++)
                cur = cur.next.merge(dir[i], new Trie(dir[i]), (o, n) -> o);
            cur.end = true;
        }
        List<String> res = new ArrayList<>();
        for (Trie nx : root.next.values())
            dfs(nx, res, "");
        return res;
    }

    public void dfs(Trie root, List<String> res, String path) {
        path += "/" + root.val;
        if (root.end) res.add(path);
        else {
            for (Trie nx : root.next.values())
                dfs(nx, res, path);
        }
    }

    public static class Trie {
        public String val;
        Map<String, Trie> next;
        public boolean end;

        public Trie(String val) {
            this.val = val;
            next = new HashMap<>();
        }
    }
    public long minCost(int[] basket1, int[] basket2) {
        Arrays.sort(basket1);
        Arrays.sort(basket2);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < basket1.length; i++) {
            map.merge(basket1[i], 1, Integer::sum);
            map.merge(basket2[i], -1, Integer::sum);
        }

        int mn = Integer.MAX_VALUE, cnt = 0;
        List<int[]> ls = new ArrayList<>();
        for (Map.Entry<Integer, Integer> e : map.entrySet()) {
            int x = e.getKey(), c = Math.abs(e.getValue());
            if (c % 2 != 0) return -1;
            mn = Math.min(mn, x);
            cnt += c / 2;
            if (c != 0) ls.add(new int[]{x, c});
        }
        ls.sort(Comparator.comparingInt(a -> a[0]));
        long res = 0;
        for (int i = 0; i < ls.size() && cnt > 0; i++) {
            int[] num = ls.get(i);
            int c = Math.min(cnt, num[1]);
            res += Math.min((long) num[0] * c / 2, (long) mn * c) ;
            cnt -= c;
        }

        return res;
    }

    private int ln;
    private int rn;
    private int xn;
    private int yn;

    public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
        // 找到 x 三个邻接点的子树结点数
        int t = dfs(root, x);
        if (root.val == x) xn = t;
        return yn >= xn || ((ln + yn >= rn + 1 || rn + yn >= ln + 1) && ln + rn >= yn + 1);
    }

    public int dfs(TreeNode root, int x) {
        if (root == null) return 0;
        int l = dfs(root.left, x);
        int r = dfs(root.right, x);
        if (x == root.val) {
            ln = l; rn = r;
        }
        else if (root.left != null && root.left.val == x) {
            xn = l; yn = r;
        }
        else if (root.right != null && root.right.val == x) {
            xn = r; yn = l;
        }
        return l + r + 1;
    }

    public long maxScore(int[] nums1, int[] nums2, int k) {
        int n = nums1.length;

        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> nums2[b] - nums2[a]);

        long max = 0, sum = 0;
        Queue<Integer> q = new PriorityQueue<>((a, b) -> a - b);

        for (int i = 0; i < n; i++) {
            int val = nums1[idx[i]];
            if (i < k) {
                q.offer(val);
                sum += val;
                if (i == k - 1) max = Math.max(max, sum * nums2[idx[i]]);
            } else if (q.peek() < val) {
                sum += (long) val - q.poll();
                q.offer(val);
                max = Math.max(max, sum * nums2[idx[i]]);
            }
        }
        return max;
    }

    public int[] shortestAlternatingPaths(int n, int[][] redEdges, int[][] blueEdges) {
        List<Integer>[][] next = new List[2][n];
        for (int i = 0; i < n; i++) {
            next[0][i] = new ArrayList<>();
            next[1][i] = new ArrayList<>();
        }
        // build graph
        for (int[] r : redEdges) next[0][r[0]].add(r[1]);
        for (int[] b : blueEdges) next[1][b[0]].add(b[1]);
        // dist array
        int[][] d = new int[2][n];
        // arrive 0 and next edge color is 0/1
        Deque<int[]> q = new ArrayDeque<>();
        q.offer(new int[]{0, 0});
        q.offer(new int[]{0, 1});
        // bfs
        while (!q.isEmpty()) {
            int[] p = q.poll();
            int x = p[0], c = p[1];
            for (int nx : next[c][x]) {
                // not visit by 'c' color edge
                if (d[c][nx] == 0 && nx != 0) {
                    d[c][nx] = d[1 - c][x] + 1;
                    q.offer(new int[]{nx, 1 - c});
                }
            }
        }
        int[] res = new int[n];
        for (int i = 1; i < n; i++) {
            if (d[0][i] == 0 && d[1][i] == 0) res[i] = -1;
            else if (d[0][i] == 0) res[i] = d[1][i];
            else if (d[1][i] == 0) res[i] = d[0][i];
            else res[i] = Math.min(d[0][i], d[1][i]);
        }

        return res;
    }

    public boolean checkXMatrix(int[][] grid) {
        int n = grid.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((j == i || j == n - i - 1) && grid[i][j] == 0)
                    return false;
                else if ((j != i && j != n - i - 1) && grid[i][j] != 0)
                    return false;
            }
        }

        return true;
    }

    public int[][] matrixRankTransform(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] r = new int[m][2];
        int[][] c = new int[n][2];

        int[][] arr = new int[m * n][3];
        for (int i = 0, idx = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                arr[idx++] = new int[]{i, j, matrix[i][j]};
        }
        Arrays.sort(arr, Comparator.comparingInt(a -> a[2]));

        uf = new int[m * n];
        for (int i = 0; i < uf.length; i++) uf[i] = i;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k1 = i + 1; k1 < m; k1++) {
                    if (matrix[i][j] == matrix[k1][j])
                        union(i * n + j, k1 * n + j);
                }
                for (int k2 = j + 1; k2 < n; k2++) {
                    if (matrix[i][j] == matrix[i][k2])
                        union(i * n + j, i * n + k2);
                }
            }
        }

        int[][] res = new int[m][n];
        int[] map = new int[m * n];
        int idx = 1;
        for (int[] p : arr) {
            int x = p[0], y = p[1], rk = 0;
            if (r[x][0] > c[y][0])
                rk = r[x][0] + (r[x][1] == p[2] ? 0 : 1);
            else if (r[x][0] < c[y][0])
                rk = c[y][0] + (c[y][1] == p[2] ? 0 : 1);
            else
                rk = r[x][0] + (Math.min(r[x][1], c[y][1]) == p[2] ? 0 : 1);

            int g = find(x * n + y);
            if (map[g] < rk) {
                map[g] = rk;
                if (idx == arr.length || p[2] != arr[idx][2]) {
                    int i = idx - 2;
                    while (i >= 0 && p[2] == arr[i][2]) {
                        int tx = arr[i][0];
                        int ty = arr[i][1];
                        res[tx][ty] = map[g];
                        r[tx][0] = res[tx][ty]; r[tx][1] = p[2];
                        c[ty][0] = res[tx][ty]; c[ty][1] = p[2];
                        i--;
                    }
                }
            }

            res[x][y] = map[g];
            r[x][0] = res[x][y]; r[x][1] = p[2];
            c[y][0] = res[x][y]; c[y][1] = p[2];
            idx++;
        }

        return res;
    }

    public int[] uf;

    public void union(int a, int b) {
        int p = find(a);
        int q = find(b);
        uf[p] = q;
    }

    public int find(int x) {
        if (uf[x] != x) uf[x] = find(uf[x]);
        return uf[x];
    }
}
