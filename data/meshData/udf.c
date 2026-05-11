#include "udf.h"

#define massflow_preheat 0.0029067
#define massflow_h2o2 0.00101573
#define massflow_drying 0.00279073

#define temperature_preheat 433.15
#define temperature_h2o2 473.15
#define temperature_drying 413.15

#define mass_fraction_h2o2 0.024989
#define mass_fraction_h2o 0.152936


DEFINE_PROFILE(inlet_a_massflow, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;
    real massflow_a;

    if (t < 0.32)
        massflow_a = 0.0;
    else if (t < 0.9)
        massflow_a = massflow_preheat;
    else if (t < 2.12)
        massflow_a = 0.0;
    else if (t < 2.7)
        massflow_a = massflow_h2o2;
    else if (t < 4.82)
        massflow_a = 0.0;
    else if (t < 5.4)
        massflow_a = massflow_drying;
    else if (t < 6.62)
        massflow_a = 0.0;
    else if (t < 7.4)
        massflow_a = massflow_drying;
	else
		massflow_a = 0.0;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = massflow_a;
    }
    end_f_loop(f, thread)
}

DEFINE_PROFILE(inlet_a_temperature, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;
    real temperature_a;

    if (t < 0.32)
        temperature_a = 300;
    else if (t < 0.9)
        temperature_a = temperature_preheat;
    else if (t < 2.12)
        temperature_a = 300;
    else if (t < 2.7)
        temperature_a = temperature_h2o2;
    else if (t < 4.82)
        temperature_a = 300;
    else if (t < 5.4)
        temperature_a = temperature_drying;
    else if (t < 6.62)
        temperature_a = 300;
    else if (t < 7.4)
        temperature_a = temperature_drying;
	else
		temperature_a = 300;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = temperature_a;
    }
    end_f_loop(f, thread)
}

DEFINE_PROFILE(inlet_a_h2o2, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;
    real h2o2_a;

	if (t < 2.12)
        h2o2_a = 0.0;
    else if (t < 2.7)
		h2o2_a = mass_fraction_h2o2;
	else
		h2o2_a = 0.0;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = h2o2_a;
    }
    end_f_loop(f, thread)
}

DEFINE_PROFILE(inlet_a_h2o, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;
    real h2o_a;

	if (t < 2.12)
        h2o_a = 0.0;
    else if (t < 2.7)
		h2o_a = mass_fraction_h2o;
	else
		h2o_a = 0.0;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = h2o_a;
    }
    end_f_loop(f, thread)
}

DEFINE_PROFILE(inlet_b_massflow, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;
    real massflow_b;

    if (t < 1.22)
        massflow_b = 0.0;
    else if (t < 1.8)
        massflow_b = massflow_preheat;
    else if (t < 3.02)
        massflow_b = 0.0;
    else if (t < 3.6)
        massflow_b = massflow_h2o2;
    else if (t < 5.72)
        massflow_b = 0.0;
    else if (t < 6.3)
        massflow_b = massflow_drying;
	else
		massflow_b = 0.0;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = massflow_b;
    }
    end_f_loop(f, thread)
}

DEFINE_PROFILE(inlet_b_temperature, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;
    real temperature_b;

    if (t < 1.22)
        temperature_b = 300.0;
    else if (t < 1.8)
        temperature_b = temperature_preheat;
    else if (t < 3.02)
        temperature_b = 300.0;
    else if (t < 3.6)
        temperature_b = temperature_h2o2;
    else if (t < 5.72)
        temperature_b = 300.0;
    else if (t < 6.3)
        temperature_b = temperature_drying;
	else
		temperature_b = 300.0;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = temperature_b;
    }
    end_f_loop(f, thread)
}

DEFINE_PROFILE(inlet_b_h2o2, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;
    real h2o2_b;

	if (t < 3.02)
        h2o2_b = 0.0;
    else if (t < 3.6)
		h2o2_b = mass_fraction_h2o2;
	else
		h2o2_b = 0.0;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = h2o2_b;
    }
    end_f_loop(f, thread)
}

DEFINE_PROFILE(inlet_b_h2o, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;
    real h2o_b;

	if (t < 3.02)
        h2o_b = 0.0;
    else if (t < 3.6)
		h2o_b = mass_fraction_h2o;
	else
		h2o_b = 0.0;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = h2o_b;
    }
    end_f_loop(f, thread)
}

