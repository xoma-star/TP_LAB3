#include <cmath>
#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;

py::object gauss_zeid_plus(py::array_t<float> u0_, py::array_t<float> u1_, py::array_t<int> red_, py::array_t<int> black_, py::array_t<float> f_, int n, int ny, int bnum, int rnum, float eps, float a1, float an, float ai) {

    py::buffer_info buf1 = u0_.request();
    py::buffer_info buf2 = u1_.request();
    py::buffer_info buf3 = red_.request();
    py::buffer_info buf4 = black_.request();
    py::buffer_info buf5 = f_.request();

    float *u0 = (float *) buf1.ptr;
    float *u1 = (float *) buf2.ptr;
    int *red = (int *) buf3.ptr;
    int *black = (int *) buf4.ptr;
    float *f = (float *) buf5.ptr;

    py::list residual;
    int iter_num = 0;
    float diff = eps + 1;
    while (diff > eps){

        diff = 0;
		// Вычисление значений в красных узлах
        for (int i=0; i<rnum; i++){
            u1[red[i]] = -(a1 * (u0[red[i] - 1] + u0[red[i] + 1]) + an * (u0[red[i] + ny] + u0[red[i] - ny]) - f[red[i]]) / ai;
        }

		//Вычисление значений в чёрных узлах
        for (int i=0; i<bnum; i++){
            u1[black[i]] = -(a1 * (u1[black[i] - 1] + u1[black[i] + 1]) + an * (u1[black[i] + ny] + u1[black[i] - ny]) - f[black[i]]) / ai;
        }

		//Вычисление разницы между текущей итерацией и следующей
        for (int i=0; i<n; i++){
            diff += abs(u0[i] - u1[i]);
        }

		//Текущая итерация принимает значение следующей
        float * temp = u0;
        u0 = u1;
        u1 = temp;

        residual.append(diff);
        diff = sqrt(diff);
        cout << diff << endl;

        iter_num++;
    }

    py::tuple result = py::make_tuple(iter_num, u0_, u1_, residual);

    return result;
};


PYBIND11_MODULE(gzWrapper,m) {
    m.doc() = "";
    m.def("GaussZeidel", &gauss_zeid_plus, "");
}