//Cycle detection in undirected graph
#include <iostream>
#include <unordered_map>
#include <list>
#include <queue>
using namespace std;

bool cycle(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,unordered_map<int,bool>& dfsvisited,int node)
{
    visited[node] = true;
    dfsvisited[node] = true;
    for(auto neighbour: adj[node]){
        if(!visited[neighbour]){
            bool ans = cycle(adj,visited,dfsvisited,neighbour);
            if(ans) return true;
        }else if(dfsvisited[neighbour]){
            return true;
        }
    }
    dfsvisited[node] = false;
    return false;
}
int main()
{
    int n; //number of edges
    cin >> n;

    unordered_map<int,list<int>> adj;
    for(int i=0;i<n;i++){
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    unordered_map<int,bool> visited;
    unordered_map<int,bool> dfsvisited;
    bool ans;
    for(int i=0;i<n;i++){
        if(!visited[i]){
            ans = cycle(adj,visited,dfsvisited,i);
            if(ans) break;
        }
    }
    if(ans) cout << "Yes"<< endl;
    else cout << "No"<< endl;
    return 0;
}