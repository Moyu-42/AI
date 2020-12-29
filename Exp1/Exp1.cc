#include <iostream>
#include <vector>
#include <queue>
#include <time.h>
#include <windows.h>
#include <memory.h>
#include <stack>
#include <algorithm>
#define A 0
#define B 1
#define C 2
#define D 3
#define E 4
#define F 5
#define G 6
#define H 7
#define I 8
#define L 9
#define M 10
#define N 11
#define O 12
#define P 13
#define R 14
#define S 15
#define T 16
#define U 17
#define V 18
#define Z 19
#define TEST_TIMES 1000
#define PII pair<int, int>
using namespace std;

int h[20] = {366, 0,   160, 242, 161, 178, 77,  151, 226, 244,
             241, 234, 380, 98,  193, 253, 329, 80,  199, 374};
string label[20] = {"Arad",    "Bucharest", "Craiova",  "Dobreta", "Eforie",
                    "Fagaras", "Giurgiu",   "Hirsova",  "Iasi",    "Lugoj",
                    "Mehadia", "Neamt",     "Oradea",   "Pitesti", "Rimnicu Vilcea",
                    "Sibiu",   "Timisoara", "Urziceni", "Vaslui",  "Zerind"};

struct node // Astar 搜索中使用的结点
{
    int g;                       // 到当前结点的路径消耗
    int h;                       // 当前结点到目标结点的预估消耗
    int f;                       // Astar的启发式函数
    int name;                    // 当前结点命名
    node(int name, int g, int h) // 结构体初始化方法
    {
        this->name = name;
        this->g = g;
        this->h = h;
        this->f = g + h;
    };
    bool operator<(const node &a) const // 运算符重载，用于排序
    {
        return f < a.f;
    }
};

class Graph {
  public:
    Graph() {
        memset(graph, -1, sizeof(graph));
    }
    int getEdge(int from, int to) { // 计算搜索算法消耗时使用的邻接矩阵
        return graph[from][to];
    }
    vector<PII> getAction(int from) { // 获得当前行动的子结点集
        return edge[from];
    }
    void addEdge(int from, int to, int cost) { // 图中添加边
        if (from >= 20 || from < 0 || to >= 20 || to < 0)
            return;
        graph[from][to] = cost;
        edge[from].push_back({to, cost});
    }
    void init() { // 图初始化
        addEdge(O, Z, 71);
        addEdge(Z, O, 71);

        addEdge(O, S, 151);
        addEdge(S, O, 151);

        addEdge(Z, A, 75);
        addEdge(A, Z, 75);

        addEdge(A, S, 140);
        addEdge(S, A, 140);

        addEdge(A, T, 118);
        addEdge(T, A, 118);

        addEdge(T, L, 111);
        addEdge(L, T, 111);

        addEdge(L, M, 70);
        addEdge(M, L, 70);

        addEdge(M, D, 75);
        addEdge(D, M, 75);

        addEdge(D, C, 120);
        addEdge(C, D, 120);

        addEdge(C, R, 146);
        addEdge(R, C, 146);

        addEdge(S, R, 80);
        addEdge(R, S, 80);

        addEdge(S, F, 99);
        addEdge(F, S, 99);

        addEdge(F, B, 211);
        addEdge(B, F, 211);

        addEdge(P, C, 138);
        addEdge(C, P, 138);

        addEdge(R, P, 97);
        addEdge(P, R, 97);

        addEdge(P, B, 101);
        addEdge(B, P, 101);

        addEdge(B, G, 90);
        addEdge(G, B, 90);

        addEdge(B, U, 85);
        addEdge(U, B, 85);

        addEdge(U, H, 98);
        addEdge(H, U, 98);

        addEdge(H, E, 86);
        addEdge(E, H, 86);

        addEdge(U, V, 142);
        addEdge(V, U, 142);

        addEdge(I, V, 92);
        addEdge(V, I, 92);

        addEdge(I, N, 87);
        addEdge(N, I, 87);
    }

  private:
    int graph[20][20];    // 邻接矩阵
    vector<PII> edge[20]; // 搜索树
};
class DFS {
  private:
    int vis[20];      // 标记结点是否被访问
    int cost;         // 搜索消耗
    vector<int> road; // 搜索过程的路径
  public:
    DFS() {
        memset(vis, 0, sizeof(vis));
        cost = 0;
        road.clear();
    }
    bool dfs_version2(
        int goal, int src, Graph &graph) { // 当前结点首先扩展其所有子结点，但只访问目前最深的子结点
        if (src == goal) { // 搜索到目标结点
            road.push_back(goal);
            return 1;
        }
        bool flag = 0; // 判断是否搜索到解
        vis[src] = 1;  // 当前结点设置为已访问
        road.push_back(src);
        stack<int> s;                           // 用于保存当前结点的所有子结点
        vector<PII> nxt = graph.getAction(src); // 获取子结点集
        for (auto w : nxt) {
            int i = w.first;
            if (vis[i] != 1)
                s.push(i); // 未被访问过，则加入子结点集
        }
        while (!s.empty()) { // 取出一个子结点
            int nxt = s.top();
            s.pop();
            int weight = graph.getEdge(src, nxt);
            cost += weight;                        // 计算消耗
            flag = dfs_version2(goal, nxt, graph); // 从当前子结点递归调用dfs
            if (flag)
                break;      // 已经搜索到目标节点，结束
            cost -= weight; // 回溯
            road.pop_back();
            vis[nxt] = 0;
        }
        return flag;
    }
    void print_result(int goal) { // 路径输出函数
        cout << "DFS"
             << ": " << endl;
        cout << "Path: ";
        for (auto v : road) {
            cout << label[v] << "->";
            if (v == goal)
                cout << "end" << endl;
        }
        cout << "Cost: " << cost << endl;
    }
};
class BFS {
  private:
    int vis[20]; // 标记结点是否被访问过
    int cost;    // 搜索消耗
    int fa[20];  // 第i个结点的父节点（用于之后路径输出）
    int flag;
    queue<int> q; // 维护队列保存已扩展结点
  public:
    BFS() {
        cost = 0;
        flag = 1;
        memset(vis, 0, sizeof(vis));
        memset(fa, -1, sizeof(fa));
        while (!q.empty())
            q.pop();
    }
    void bfs(int goal, int src, Graph &graph) {
        vis[src] = 1;            // 源节点设置为已访问
        q.push(src);             // 源节点入队
        while (!q.empty()) {     // 从队列中取元素直到队列为空或搜索到目标节点
            int now = q.front(); // 取队首元素
            q.pop();
            if (now == goal) { // 找到目标节点，结束
                flag = 0;
                break;
            }
            vector<PII> nxt = graph.getAction(now); // 获取子结点
            for (auto w : nxt) {
                int weight = w.second;
                int i = w.first;
                if (vis[i] == 0) { // 未放问过
                    q.push(i);     // 入队
                    vis[i] = 1;    // 设置为已访问
                    fa[i] = now;
                }
            }
        }
    }
    void print_result(int goal, Graph &graph) { // 路径输出及消耗计算函数，当然这不重要
        cout << "BFS: " << endl;
        if (flag) { // 未能找到目标结点
            cout << "Failed" << endl;
            return;
        }
        cout << "Path: ";
        stack<int> s; // 维护访问过的结点集合，从后向前
        s.push(goal); 
        int father = fa[goal]; // 寻找父结点
        cost += graph.getEdge(father, goal); // 计算路径消耗
        while (father != -1) { // 直到找到根节点结束
            s.push(father);
            cost += graph.getEdge(father, fa[father]);
            father = fa[father];
        }
        while (!s.empty()) { // 路径输出
            cout << label[s.top()] << "->";
            s.pop();
        }
        cout << "end" << endl;
        cout << "Cost: " << cost << endl;
    }
};
class UCS {
  private:
    int cost;
    priority_queue<PII, vector<PII>, greater<PII>> q; // 优先队列保存结点
    int fa[20];
    int dis[20];
    int flag;
  public:
    UCS() {
        cost = 0;
        flag = 1;
        while (!q.empty())
            q.pop();
        memset(dis, 0x3f3f3f3f, sizeof(dis)); // 初始化为极大值INF
        memset(fa, -1, sizeof(fa));
    }
    void ucs(int goal, int src, Graph &graph) {
        q.push((PII){0, src}); // 源节点入队，用于比较的键值为当前节点的消耗
        while (!q.empty()) {
            PII now = q.top(); // 取队首
            q.pop();
            int from = now.second; // 把元素取出来只是为了方便后面写
            int cost_ = now.first;
            if (from == goal) { // 到达目标节点
                flag = 0;
                break;
            }
            vector<PII> nxt = graph.getAction(from); // 获取子结点集
            for (auto w : nxt) {
                int weight = w.second;
                int i = w.first;
                q.push((PII){cost_ + weight, i}); // 入队
                if (dis[i] > cost_ + weight) { // 如果当前代价更新后最优，更新其父结点为当前结点
                    dis[i] = min(dis[i], cost_ + weight);
                    fa[i] = from;
                }
            }
        }
    }
    void print_result(int goal, Graph &graph) { // 同BFS
        cout << "UCS: " << endl;
        if (flag) {
            cout << "Failed" << endl;
            return;
        }
        cout << "Path: ";
        stack<int> s;
        s.push(goal);
        int father = fa[goal];
        cost += graph.getEdge(father, goal);
        while (father != A) {
            s.push(father);
            cost += graph.getEdge(father, fa[father]);
            father = fa[father];
        }
        s.push(A);
        while (!s.empty()) {
            cout << label[s.top()] << "->";
            s.pop();
        }
        cout << "end" << endl;
        cout << "Cost: " << cost << endl;
    }
};
class IDS {
  private:
  private:
    int vis[20];
    int cost;
    int iter;
    vector<int> road;
  public:
    IDS() {
        memset(vis, 0, sizeof(vis));
        cost = 0;
        iter = 1;
        road.clear();
    }
    void ids(int goal, int src, Graph &graph) {
        for (; iter < 1000; iter++) { // 设置最大迭代次数
            memset(vis, 0, sizeof(vis));
            road.clear();
            cost = 0;
            if (dfs(goal, src, graph, 1, iter) == 1)
                break; // 搜索到解，结束算法
        }
    }
    int dfs(int goal, int src, Graph &graph, int it, int max_depth) {
        if (src == goal) { // 搜索到目标节点，返回 1
            road.push_back(goal);
            return 1;
        }
        int flag = 0;
        road.push_back(src);
        vis[src] = 1; // 设置当前结点已访问
        if (it == max_depth)
            return 2;                           // 到达最大深度，返回 2
        stack<int> s;                           // 保存待扩展子结点集
        vector<PII> nxt = graph.getAction(src); // 获取子结点集
        for (auto w : nxt) {                    // 同DFS
            int weight = w.second;
            int i = w.first;
            if (vis[i] != 1)
                s.push(i);
        }
        while (!s.empty()) { // 大致同DFS
            int nxt = s.top();
            s.pop();
            int weight = graph.getEdge(src, nxt);
            cost += weight;
            flag = dfs(goal, nxt, graph, it + 1, max_depth);
            if (flag == 1)
                break; // 搜索到解，结束；否则进行回溯
            cost -= weight;
            road.pop_back();
            vis[nxt] = 0;
        }
        return flag;
    }
    void print_result(int goal) { // 同DFS
        cout << "IDS"
             << ": " << endl;
        for (int i = 1; i < iter; i++)
            cout << "Iter " << i - 1 << " Failed" << endl;
        cout << "Path: ";
        for (auto v : road) {
            cout << label[v] << "->";
            if (v == goal)
                cout << "end" << endl;
        }
        cout << "Cost: " << cost << endl;
    }
};
class GREEDY {
  private:
    int cost;
    priority_queue<PII, vector<PII>, greater<PII>> q; // 优先队列存放子结点
    int vis[20];
    int fa[20];
    int flag;

  public:
    GREEDY() {
        cost = 0;
        flag = 1;
        while (!q.empty())
            q.pop();
        memset(fa, -1, sizeof(fa));
        memset(vis, 0, sizeof(vis));
    }
    void greedy(int goal, int src, Graph &graph) {
        q.push((PII){h[src], src}); // 源节点入队，用于比较的键值为估计的到达目标节点的距离
        vis[src] = 1;
        while (!q.empty()) {
            PII now = q.top();
            q.pop();
            int from = now.second;
            int cost_ = now.first;
            if (from == goal) {
                flag = 0;
                break;
            }
            vector<PII> nxt = graph.getAction(from); // 获取子结点集
            for (auto w : nxt) {
                int weight = w.second;
                int i = w.first;
                if (weight != -1 && vis[i] == 0) {
                    q.push((PII){h[i], i}); // 结点未访问过，入队
                    fa[i] = from;           // 设置父结点
                    vis[i] = 1;
                }
            }
        }
    }
    void print_result(int goal, Graph &graph) {
        cout << "GREEDY: " << endl;
        if (flag) {
            cout << "Failed" << endl;
            return;
        }
        cout << "Path: ";
        stack<int> s;
        s.push(goal);
        int father = fa[goal];
        cost += graph.getEdge(father, goal);
        while (father != A) {
            s.push(father);
            cost += graph.getEdge(father, fa[father]);
            father = fa[father];
        }
        s.push(A);
        while (!s.empty()) {
            cout << label[s.top()] << "->";
            s.pop();
        }
        cout << "end" << endl;
        cout << "Cost: " << cost << endl;
    }
};
class AStar {
  private:
    bool list[20];
    vector<node> openList; // 模拟优先队列
    bool closeList[20];    // 当前结点是否不可再访问
    stack<int> road;
    int parent[20];

  public:
    AStar() {
        memset(parent, -1, sizeof(parent));
        list[A] = true;
        memset(closeList, 0, sizeof(closeList));
    }
    void A_star(int goal, node &src, Graph &graph) {
        openList.push_back(src);                // 源节点入队
        sort(openList.begin(), openList.end()); // 按启发式函数排序

        while (!openList.empty()) // 队列非空，继续
        {
            node curr = openList[0]; // 取队首元素

            if (curr.name == B)
                break; // 搜索到目标节点，结束

            openList.erase(openList.begin()); // 队首元素出队
            if (closeList[curr.name])
                continue;                                 // 如果在closelist内，继续循环
            closeList[curr.name] = 1;                     // 将当前结点放入closelist内
            vector<PII> nxt = graph.getAction(curr.name); // 获取子结点集
            for (auto w : nxt) {
                int weight = w.second;
                int i = w.first;
                if (closeList[i] == 0) { // 不在closelist内
                    bool flag = 1;
                    for (auto it = openList.begin(); it != openList.end(); it++) {
                        // 如果队列中存在该元素，判断是否需要对其更新：当前函数值更优则更新
                        if (it->name == i) {               // 在队列中
                            if (curr.g + weight < it->g) { // 更优
                                openList.erase(it);        // 移除队列中原有的该结点
                                break;
                            } else {
                                flag = 0;
                                break;
                            }
                        }
                    }
                    if (flag) { // 新的结点入队
                        node nxt(i, curr.g + weight, h[i]);
                        openList.push_back(nxt);
                        parent[i] = curr.name; // 设置父结点
                    }
                }
            }
            sort(openList.begin(), openList.end()); // 排序，模拟堆
        }
    }
    void print_result(Graph &graph) // 打印输出
    {
        int p = openList[0].name;
        int lastNodeNum;
        road.push(p);
        while (parent[p] != -1) {
            road.push(parent[p]);
            p = parent[p];
        }
        lastNodeNum = road.top();
        int cost = 0;
        cout << "A*: " << endl;
        cout << "Path: ";
        while (!road.empty()) {
            cout << label[road.top()] << "->";
            if (road.top() != lastNodeNum) {
                cost += graph.getEdge(lastNodeNum, road.top());
                lastNodeNum = road.top();
            }
            road.pop();
        }
        cout << "end" << endl;
        cout << "Cost:" << cost << endl;
    }
};

int main() {
    freopen("output.txt", "w", stdout); // 输出文件保存在output.txt内，如果要在终端显示请注释该行
    Graph graph;
    graph.init();
    node src(A, 0, h[0]);
    int goal = B;

    double run_time;
    _LARGE_INTEGER time_start;
    _LARGE_INTEGER time_over;
    double dqFreq;
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    dqFreq = (double)f.QuadPart;
    /* 以下为程序运行部分，首先运行一次获取输出，之后测试1000次，对运行时间取平均值，用于时间测试 */
    // 运行BFS
    BFS b;
    b.bfs(goal, A, graph);
    b.print_result(goal, graph);

    run_time = 0;
    for (int it = 0; it < TEST_TIMES; it++) {
        BFS b;
        QueryPerformanceCounter(&time_start);
        b.bfs(goal, A, graph);
        QueryPerformanceCounter(&time_over);
        run_time += (time_over.QuadPart - time_start.QuadPart) / dqFreq;
    }
    printf("time: %.8lfms\n\n", run_time / TEST_TIMES * 1000);
    // 运行DFS
    DFS d;
    d.dfs_version2(goal, A, graph);
    d.print_result(goal);

    run_time = 0;
    for (int it = 0; it < TEST_TIMES; it++) {
        DFS d;
        QueryPerformanceCounter(&time_start);
        d.dfs_version2(goal, A, graph);
        QueryPerformanceCounter(&time_over);
        run_time += (time_over.QuadPart - time_start.QuadPart) / dqFreq;
    }
    printf("time: %.8lfms\n\n", run_time / TEST_TIMES * 1000);
    // 运行一致代价搜索
    UCS u;
    u.ucs(goal, A, graph);
    u.print_result(goal, graph);

    run_time = 0;
    for (int it = 0; it < TEST_TIMES; it++) {
        UCS u;
        QueryPerformanceCounter(&time_start);
        u.ucs(goal, A, graph);
        QueryPerformanceCounter(&time_over);
        run_time += (time_over.QuadPart - time_start.QuadPart) / dqFreq;
    }
    printf("time: %.8lfms\n\n", run_time / TEST_TIMES * 1000);
    // 运行迭代加深搜索
    IDS i;
    i.ids(goal, A, graph);
    i.print_result(goal);

    run_time = 0;
    for (int it = 0; it < TEST_TIMES; it++) {
        IDS i;
        QueryPerformanceCounter(&time_start);
        i.ids(goal, A, graph);
        QueryPerformanceCounter(&time_over);
        run_time += (time_over.QuadPart - time_start.QuadPart) / dqFreq;
    }
    printf("time: %.8lfms\n\n", run_time / TEST_TIMES * 1000);
    // 运行贪婪搜素
    GREEDY g;
    g.greedy(goal, A, graph);
    g.print_result(goal, graph);

    run_time = 0;
    for (int it = 0; it < TEST_TIMES; it++) {
        GREEDY g;
        QueryPerformanceCounter(&time_start);
        g.greedy(goal, A, graph);
        QueryPerformanceCounter(&time_over);
        run_time += (time_over.QuadPart - time_start.QuadPart) / dqFreq;
    }
    printf("time: %.8lfms\n\n", run_time / TEST_TIMES * 1000);
    // 运行AStar搜素
    AStar a;
    a.A_star(goal, src, graph);
    a.print_result(graph);

    run_time = 0;
    for (int it = 0; it < TEST_TIMES; it++) {
        AStar a;
        QueryPerformanceCounter(&time_start);
        a.A_star(goal, src, graph);
        QueryPerformanceCounter(&time_over);
        run_time += (time_over.QuadPart - time_start.QuadPart) / dqFreq;
    }
    printf("time: %.8lfms\n\n", run_time / TEST_TIMES * 1000);
}
