#include <iostream>
#include <list>
#include <unordered_map>
#include <queue>
#include <stack>
using namespace std;

int main()
{
    int e;
    cout << "Number of edges" << endl;
    cin >> e;
    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;
    unordered_map<int,int> parent; //helpfule in backtrack

    //we are talking about undirected graph
    for(int i=0;i<e;i++){
        int u,x;
        cin >> u >> x;
        adj[u].push_back(x);
        adj[x].push_back(u);
    }
    cout << "Please enter your source and target node" << endl;
    int src,target;
    cin >> src>> target;
    queue<int> q;
    q.push(src);
    visited[src] = true;
    parent[src] = -1;
    while (!q.empty())
    {
        int front = q.front();
        q.pop();

        for(auto i:adj[front]){
            if(!visited[i]){
                q.push(i);
                visited[i]  = true;
                parent[i] = front;
            }
        }
    }

    vector<int> ans;
    int curr = target;
    ans.push_back(target);
    while (curr!=src)
    {
        curr = parent[curr];
        ans.push_back(curr);
    }
    
    for(auto i : ans) cout << i << " ";
    cout << endl;
    return 0;
}