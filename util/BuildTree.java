package util;

import tree.TreeNode;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BuildTree {

    private final static Map<String, Node> map = new HashMap<>();

    public static class Node {
        protected String val;
        protected List<Node> next;

        public Node(String val, List<Node> next) {
            this.val = val;
            this.next = next;
        }
    }

    public static TreeNode buildTree(String str) {
        String[] nodeVals = str.substring(1, str.length() - 1).split(",");
        TreeNode[] nodes = new TreeNode[nodeVals.length];

        for (int i = 0; i < nodeVals.length; i++) {
            if (nodeVals[i].equals("null")) continue;
            TreeNode node = new TreeNode(Integer.parseInt(nodeVals[i]));
            nodes[i] = node;
            if (i == 0) continue;
            if ((i & 1) == 1)
                nodes[i - 1 >> 1].left = node;
            else
                nodes[i - 1 >> 1].right = node;
        }

        return nodes[0];
    }

    public static void main(String[] args) {
        TreeNode root = buildTree("[1,2,-3,-5,null,4,null]");
        System.out.println(root);
    }

//    public static void main(String[] args) {
//        File file = new File("tree.txt");
//        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
//            String line;
//            while ((line = br.readLine()) != null) {
//                String[] nd = line.split("&&");
//                Node node = map.get(nd[0]);
//                if (node == null) {
//                    node = new Node(line, new ArrayList<>());
//                    map.put(nd[0], node);
//                }
//                Node next = map.get(nd[1]);
//                if (next == null) {
//                    next = new Node(line, new ArrayList<>());
//                    map.put(nd[1], next);
//                }
//                node.next.add(next);
//            }
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//    }
}
