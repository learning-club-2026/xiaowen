#include <iostream>
#include <vector>
using namespace std;

vector<vector<double>> matMultMat(vector<vector<double>> a, vector<vector<double>> b) {
    int a_row = a.size();
    int a_col = a[0].size();
    int b_row = b.size();
    int b_col = b[0].size();

    if (a_col != b_row) {
        cout << "错误" << endl;
        return {};
    }

    vector<vector<double>> res(a_row, vector<double>(b_col, 0.0));
    for (int i = 0; i < a_row; i++) {
        for (int j = 0; j < b_col; j++) {
            for (int k = 0; k < a_col; k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}

int main() {
    int r1, c1;
    cout << "输入第一个矩阵 行数 列数：";
    cin >> r1 >> c1;
    vector<vector<double>> mat1(r1, vector<double>(c1));
    cout << "输入第一个矩阵数据：" << endl;
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
            cin >> mat1[i][j];
        }
    }

    int r2, c2;
    cout << "输入第二个矩阵 行数 列数：";
    cin >> r2 >> c2;
    vector<vector<double>> mat2(r2, vector<double>(c2));
    cout << "输入第二个矩阵数据：" << endl;
    for (int i = 0; i < r2; i++) {
        for (int j = 0; j < c2; j++) {
            cin >> mat2[i][j];
        }
    }

    vector<vector<double>> result = matMultMat(mat1, mat2);
    cout << "结果：" << endl;
    for (int i = 0; i < result.size(); i++) { 
      for (int j = 0; j < result[i].size(); j++) { 
        cout << result[i][j] << " "; 
    }
    cout << endl;
}
    return 0;
}
