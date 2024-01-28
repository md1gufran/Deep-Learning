//shortest distance between the source nodes and all other nodes
// of DAG-> Directed acyclic graph
#include <iostream>
#include <list>
#include <limits.h>
#include <stack>
#include <queue>
#include <unordered_map>
#include <vector>
using namespace std;

void dfs(unordered_map<int,list<pair<int,int>>>& adj,unordered_map<int,bool>& visited,int node,stack<int>& st)
{
    visited[node] = true;

    for(auto i: adj[node]){
        if(!visited[i.first]){
            dfs(adj,visited,i.first,st);
        }
    }
    st.push(node);
}

//shortest from the given source node with all the nodes
void shortestpath(vector<int>& dis,unordered_map<int,bool>& visited,int src,unordered_map<int,list<pair<int,int>>>& adj,stack<int>& st)
{
    dis[src] = 0;

    while(!st.empty()){
        int top = st.top();
        st.pop();

        if(dis[top] != INT_MAX){
            for(auto i: adj[top]){
                if(dis[top] + i.second < i.first){
                    dis[i.first] = dis[top] + i.second; 
                }
            }
        }
    }
}

int main()
{
    int e,v;
    cin >> e >> v;
    unordered_map<int,list<pair<int,int>>> adj;
    for(int i=0;i<e;i++){
        int u,x,weight;
        cin >> u >> x >> weight;
        adj[u].push_back(make_pair(x,weight));
    }

    unordered_map<int,bool> visited;
    stack<int> st;
    for(int i=0;i<v;i++){
        if(!visited[i]){
            dfs(adj,visited,i,st);
        }
    }

    vector<int> dis(v,INT_MAX);

    for(auto i: dis) cout << i << " ";
    cout << endl;
    return 0;
}