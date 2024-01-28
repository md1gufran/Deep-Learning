#include <iostream>
#include <list>
#include <unordered_map>
#include <queue>
#include <stack>
using namespace std;

void topsort(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,int node,stack<int>& st)
{
    visited[node] = true;

    for(auto i: adj[node]){
        if(!visited[i]){
            topsort(adj,visited,i,st);
        }
    }
    st.push(node);
}
int main()
{
    int e,v;
    cin >> e >> v;
    
    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;

    for(int i=0;i<e;i++){
        int u,x;
        cin >> u >> x;
        adj[u].push_back(x);
    }

    stack<int> st;
    for(int i=0;i<v;i++){
        if(!visited[i]){
            topsort(adj,visited,i,st);
        }
    }

    while (!st.empty())
    {
        cout << st.top() << " ";
        st.pop();
    }
    
}