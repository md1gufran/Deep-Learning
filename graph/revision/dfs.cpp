#include <iostream>
#include <list>
#include <queue>
#include <vector>
#include <unordered_map>
using namespace std;

void dfs(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,int node)
{   
    if(!visited[node]) cout << node << " ";
    visited[node] = true;

    //calling dfs for all of its neighbour one by one
    for(auto i:adj[node]){
        if(!visited[i]){
            dfs(adj,visited,i);
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
        cin >> u >> x;
        adj[u].push_back(x);
    }

    //for undirected graph
    for(int i=0;i<e;i++){
        int u,x;
        cin >> u >> x;
        adj[u].push_back(x);
        adj[x].push_back(u);
    }

    for(int i=0;i<v;i++){
        if(!visited[i]){
            dfs(adj,visited,i);
        }
    }
    return 0;
}