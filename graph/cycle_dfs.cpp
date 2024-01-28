#include <iostream>
#include <unordered_map>
#include <list>
#include <queue>
using namespace std;

bool iscyclic(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,int node,int parent)
{
    visited[node] = 1;

    for(auto neighbour : adj[node]){
        if(!visited[neighbour]){
            bool cycle = iscyclic(adj,visited,neighbour,node);
            if(cycle) return true;
        }else if( neighbour != parent){
            return true;
        }
    }
    return false;
}
int main()
{
    int n;  //number of edges
    cin >> n;

    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;
    bool ans;

    for(int i=0;i<n;i++){
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    for(int i=0;i<n;i++){
        if(!visited[i]){
            ans = iscyclic(adj,visited,i,-1);
        }
        if(ans) break;
    }

    if(ans) cout << "Yes" << endl;
    else cout << "No" << endl;
}