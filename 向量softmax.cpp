#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

vector<double> softmax(vector<double> x)
{
    double sum = 0;
    for (int i = 0; i < x.size(); i++)
        sum += exp(x[i]);

    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++)
        res[i] = exp(x[i]) / sum;

    return res;
}

int main()
{
    int n;
    cin >> n;
    vector<double> v(n);
    for (int i = 0; i < n; i++)
        cin >> v[i];

    vector<double> ans = softmax(v);
    for (int i = 0; i < ans.size(); i++)
        cout << ans[i] << " ";

    return 0;
}
