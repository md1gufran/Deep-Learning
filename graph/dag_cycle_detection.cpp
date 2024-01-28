//cycle detection with the help of topological sort in dag graph
// logic is that is a cycle is present in topological sort than
// we will have invalid top sort in bfs traversal
// means if is does not contin cycle than top sort is valid and we can visit all the nodes
// but if there is a cycle than it will not iterate through all the nodes
// that is cathc in it
#include <iostream>
#include <list>
#include <queue>
#include <vector>
#include <unordered_map>
using namespace std;

//cycle detection in DAG using bfs
int main()
{
    int e,v;
    cout << "Please enter number of edges" <<endl;
    cin >> e ;
    cout << "Please enter number of vertex" << endl;
    cin >> v;

    unordered_map<int,list<int>> adj;
    for(int i=0;i<e;i++){
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    vector<int> indegree(v);
    for(auto i: adj){
        for(auto j:i.second){
            indegree[j]++;
        }
    }

    int cnt = 0;
    queue<int> q;
    for(int i=0;i<v;i++){
        if(indegree[i]==0){
            q.push(i);
        }
    }

    while (!q.empty())
    {
        int front = q.front();
        q.pop();
        cnt++;

        for(auto i:adj[front]){
            indegree[i]--;
            if(indegree[i]==0)  
                q.push(i);
        }
    }
    
    bool ans = (cnt==v?true:false);
    if(ans) cout << "No cycle is present" << endl;
    else cout << "Yes cycle is there" << endl;
    return 0;
}