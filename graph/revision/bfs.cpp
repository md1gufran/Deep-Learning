#include <iostream>
#include <list>
#include <queue>
#include <vector>
#include <unordered_map>
using namespace std;

void bfs(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,int node)
{
    queue<int> q;
    q.push(node);
    visited[node] = true;
    while (!q.empty())
    {
        int front = q.front();
        q.pop();
        cout << front << " ";
        for(auto i :adj[front]){
            if(!visited[i]){
                q.push(i);
                visited[i] = true;
            }
        }
    }
}
int main()
{
    int e,v;
    cin >> e >> v;
    
    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;

    //for directed graph
    for(int i=0;i<e;i++){
        int u,x;
        adj[u].push_back(x);
    }   

    for(int i=0;i<e;i++){
        int u,x;
        cin >> u >> x;
        adj[u].push_back(x);
        adj[x].push_back(v);
    }
    
    for(int i=0;i<v;i++){
        if(!visited[i]){
            bfs(adj,visited,i);
        }
    }
    return 0;
}