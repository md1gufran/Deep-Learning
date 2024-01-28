#include <iostream>
#include <list>
#include <unordered_map>
#include <queue>
using namespace std;

bool dfscycle(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,unordered_map<int,bool>& dfsvisited,int node)
{
    visited[node] = true;
    dfsvisited[node] = true;

    for(auto i:adj[node]){
        if(!visited[i]){
            bool ans = dfscycle(adj,visited,dfsvisited,i);
            if(ans) return true;
        }else if(dfsvisited[i]){
            return true;
        }
    }
    dfsvisited[node] = false;
    return false;
}
int main()
{
    int e,v;
    cin >> e>> v;

    unordered_map<int,list<int>>adj;
    unordered_map<int,bool> visited;
    unordered_map<int,bool> dfsvisited;

    //in drected graph
    for(int i=0;i<e;i++){
        int u,x;
        cin >> u >> x;
        adj[u].push_back(x);
    }

}