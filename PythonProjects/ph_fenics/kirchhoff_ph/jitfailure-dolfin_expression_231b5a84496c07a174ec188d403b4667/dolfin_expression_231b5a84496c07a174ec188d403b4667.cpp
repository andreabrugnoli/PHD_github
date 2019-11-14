
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
  class dolfin_expression_231b5a84496c07a174ec188d403b4667 : public Expression
  {
     public:
       double w_0;
double Lx;
double Ly;
double beta;
double t;


       dolfin_expression_231b5a84496c07a174ec188d403b4667()
       {
            _value_shape.push_back(2);
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = w_0/(pow(Lx*Ly, 4))*(pow(Lx,2)/2*x[0] - 3*Lx*pow(x[0], 2) + 4*pow(x[0],3))*pox(Ly/2-x[1], 4)*sin(beta*t);
          values[1] = -4*w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 3)*sin(beta*t);

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "w_0") { w_0 = _value; return; }          if (name == "Lx") { Lx = _value; return; }          if (name == "Ly") { Ly = _value; return; }          if (name == "beta") { beta = _value; return; }          if (name == "t") { t = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "w_0") return w_0;          if (name == "Lx") return Lx;          if (name == "Ly") return Ly;          if (name == "beta") return beta;          if (name == "t") return t;
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

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_231b5a84496c07a174ec188d403b4667()
{
  return new dolfin::dolfin_expression_231b5a84496c07a174ec188d403b4667;
}

