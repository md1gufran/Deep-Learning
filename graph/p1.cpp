#include <iostream>
#include <unordered_map>
#include <list>
#include <queue>
using namespace std;


//This is helpful when we have a graph which dont have such dfferent component
// So in just one go we can traverse all the graph
void bfs(unordered_map<int,list<int>>& adj,vector<int>& ans,unordered_map<int,bool>& visited)
{
    queue<int> q;
    q.push(1);
    visited[1] = true;
    while(!q.empty()){
        int front = q.front();
        q.pop();

        ans.push_back(front);
        for(auto i: adj[front]){
            if(!visited[i]){
                q.push(i);
                visited[i] = true;
            }
        }
    }
}
int main(){
    int n;  //number of vertex
    cin >> n;
    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;
    vector<int> ans;
    for(int i=0;i<n;i++){
        int u ,v;
        cin >> u >> v;
        //assuming it to be directed graph
        adj[u].push_back(v);
        visited[u] = false;
    }

    bfs(adj,ans,visited);
    for(auto i : ans) cout << i << " ";
}