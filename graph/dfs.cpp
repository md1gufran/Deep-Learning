#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
using namespace std;

void dfs(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,int node,vector<int>& component)
{
    component.push_back(node);
    visited[node] = true;

    for(auto i: adj[node]){
        if(!visited[i]){
            dfs(adj,visited,i,component);
        }
    }
}
int main()
{
    int n;  //number of nodes
    cin >> n;
    
    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;

    for(int i=0;i<n;i++){
        int u,v;
        cin >> u >> v;

        adj[u].push_back(v);
        adj[v].push_back(u);
        visited[u] = false;
        visited[v] = false;
    }

    //If there any kind of component
    for(int i=0;i<n;i++){
        if(!visited[i]){
            vector<int> component;
            dfs(adj,visited,i,component);
            for(auto i:component) cout << i << " ";
            cout << endl;
        }

    }

    return 0;
}
