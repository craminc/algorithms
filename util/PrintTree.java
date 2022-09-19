package util;

import java.util.ArrayDeque;
import java.util.Deque;

public class PrintTree {

    public static class Node<T> {
        public T val;
        public Node<T> left;
        public Node<T> right;
    }

    public static <T> void print(T[] array) {
        System.out.println("-----------------------------------------");
        Deque<Integer> queue = new ArrayDeque<>();
        queue.offer(1);
        int curLevel = 1;
        int nextLevel = 0;
        int level = 1;
        while (!queue.isEmpty()) {
            Integer idx = queue.poll();
            int i = idx - 1;
            if (i >= array.length) continue;
            if (array[i] == null || (Integer) array[i] == 0) {
                System.out.print("   ");
                continue;
            }
            int k = array.length / level;
            while (k-- > 0) System.out.print(" ");
            System.out.printf("%3s ", array[i]);
            queue.offer(idx * 2);
            queue.offer(idx * 2 + 1);
            curLevel--;
            nextLevel += 2;
            if (curLevel == 0) {
                System.out.println();
                curLevel = nextLevel;
                nextLevel = 0;
                level <<= 1;
            }
        }
        System.out.println();
        System.out.println("-----------------------------------------");
    }
    public static <T> void print(Node<T> root) {
        Deque<Node<T>> queue = new ArrayDeque<>();
        queue.offer(root);
        int curLevel = 1;
        int nextLevel = 0;
        while (!queue.isEmpty()) {
            Node<T> node = queue.poll();
            if (node == null) {
                System.out.print("   ");
                continue;
            }
            System.out.printf("%3s ", node.val);
            queue.offer(node.left);
            queue.offer(node.right);
            curLevel--;
            nextLevel += 2;
            if (curLevel == 0) {
                System.out.println();
                curLevel = nextLevel;
                nextLevel = 0;
            }
        }
    }

    public static void main(String[] args) {
        print(new Integer[]{8, 8, 5, 8, 6, 4, 5, 1, 8, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    }
}
