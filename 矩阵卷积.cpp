#include <iostream>
#include <vector>
using namespace std;

typedef vector<vector<double>> Matrix;

Matrix convolve(const Matrix& input, const Matrix& kernel)
{
    int input_h = input.size();
    int input_w = input[0].size();
    int k_h = kernel.size();
    int k_w = kernel[0].size();
    
    int out_h = input_h - k_h + 1;
    int out_w = input_w - k_w + 1;
    Matrix output(out_h, vector<double>(out_w, 0.0));

    for(int i=0; i<out_h; i++)
    {
        for(int j=0; j<out_w; j++)
        {
            double sum = 0;
            for(int k=0; k<k_h; k++)
            {
                for(int l=0; l<k_w; l++)
                {
                    sum += input[i+k][j+l] * kernel[k][l];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

int main()
{
    int h, w;
    cout << "矩阵行数 列数：";
    cin >> h >> w;
    Matrix input(h, vector<double>(w));
    for(int i=0; i<h; i++)
        for(int j=0; j<w; j++)
            cin >> input[i][j];

    int kh, kw;
    cout << "卷积核行数 列数：";
    cin >> kh >> kw;
    Matrix kernel(kh, vector<double>(kw));
    for(int i=0; i<kh; i++)
        for(int j=0; j<kw; j++)
            cin >> kernel[i][j];

    Matrix res = convolve(input, kernel);

    for(int i=0; i<res.size(); i++)
    {
        for(int j=0; j<res[i].size(); j++)
        {
            cout << res[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
