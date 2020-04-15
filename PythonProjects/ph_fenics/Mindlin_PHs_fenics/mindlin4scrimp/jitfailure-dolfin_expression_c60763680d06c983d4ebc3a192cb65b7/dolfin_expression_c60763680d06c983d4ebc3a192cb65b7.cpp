
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_c60763680d06c983d4ebc3a192cb65b7 : public Expression
  {
     public:
       double x0;
double xL;
double y0;
double yL;
double ampl;
double sX;
double sY;
double X0;
double Y0;
double rho;


       dolfin_expression_c60763680d06c983d4ebc3a192cb65b7()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = gau_Apw_0;

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "x0") { x0 = _value; return; }          if (name == "xL") { xL = _value; return; }          if (name == "y0") { y0 = _value; return; }          if (name == "yL") { yL = _value; return; }          if (name == "ampl") { ampl = _value; return; }          if (name == "sX") { sX = _value; return; }          if (name == "sY") { sY = _value; return; }          if (name == "X0") { X0 = _value; return; }          if (name == "Y0") { Y0 = _value; return; }          if (name == "rho") { rho = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "x0") return x0;          if (name == "xL") return xL;          if (name == "y0") return y0;          if (name == "yL") return yL;          if (name == "ampl") return ampl;          if (name == "sX") return sX;          if (name == "sY") return sY;          if (name == "X0") return X0;          if (name == "Y0") return Y0;          if (name == "rho") return rho;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_c60763680d06c983d4ebc3a192cb65b7()
{
  return new dolfin::dolfin_expression_c60763680d06c983d4ebc3a192cb65b7;
}

