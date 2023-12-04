#include <math.h>
#include <random>
#include <complex>
#include <limits>
#include <ostream>
#include <type_traits>
#include <fenv.h>
#pragma STDC FENV_ACCESS ON



//this file contains stochastic +, - ,* and /. They need to be called explicitly by sadd() ssub()
//smult() and sdiv() respectively

float RoundAway(double db) {
    float to_ret = 0.0;
    if(db < 0) {
        const int originalRounding = fegetround();
        fesetround(FE_DOWNWARD);
        to_ret = float(db);
        fesetround(originalRounding);
    } else if(db > 0) {
        const int originalRounding = fegetround();
        fesetround(FE_UPWARD);
        to_ret = float(db);
        fesetround(originalRounding);
    }
    return to_ret;
}

 float Stochastic_Round(double db) {
   const int originalRounding = fegetround();
   fesetround(FE_TOWARDZERO);
   float RZ = float(db);
   fesetround(originalRounding);
   float RA = RoundAway(db);
   std::uniform_real_distribution<double> unif(0,1);
   std::default_random_engine re;
   double samp = unif(re);
   double q = (db - double(RZ))/(double(RA) - double(RZ));
   if(samp >= q) return RZ;
   else return RA;
 }

 inline float sadd(float a, float b, bool on)
 {
    return on? Stochastic_Round(double(a) + double(b)) : a+b;
 }
 inline float smult(float a, float b, bool on)
 {
    return on ? Stochastic_Round(double(a) * double(b)) :a*b;
 }
 inline float sdiv(float a, float b, bool on)
 {
    return on ? Stochastic_Round(double(a) / double(b)) : a/b;
 }
 inline float ssub(float a, float b, bool on)
 {
    return on? Stochastic_Round(double(a) - double(b)) : a-b ;
 }