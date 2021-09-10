:Reference :Hu et al., Nature Neuroscience, 2009

: Adapted by Mickael Zbili @ BBP, 2020:
: LJP: corrected in the paper

NEURON	{
	SUFFIX Na12Hu2009
	USEION na READ ena WRITE ina
	RANGE gNa12bar, gNa12, ina, vshifth, vshiftm, slopeh, slopem
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNa12bar = 0.00001 (S/cm2)
	vshifth = 0 (mV)
	vshiftm = 6 (mV)
	slopeh = 5
	slopem = 7
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNa12	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNa12 = gNa12bar*m*m*m*h
	ina = gNa12*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((34-23)/10)

  UNITSOFF
    if(v == (-43+vshiftm)){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- (-43+vshiftm)))/(1-(exp(-(v- (-43+vshiftm))/slopem)))
		mBeta  = (0.124 * (-v + (-43+vshiftm)))/(1-(exp(-(-v + (-43+vshiftm))/slopem)))
		mTau = (1/(mAlpha + mBeta))/qt
		mInf = (mAlpha/(mAlpha + mBeta))

    if(v == (-50+vshifth)){
      v = v + 0.0001
    }
		hAlpha = (0.024 * (v- (-50+vshifth)))/(1-(exp(-(v- (-50+vshifth))/slopeh)))
        
    if(v == (-75+vshifth)){
      v = v + 0.0001
    }
		hBeta  = (0.0091 * (-v +(-75+vshifth)))/(1-(exp(-(-v +(-75+vshifth))/slopeh)))
		hTau = (1/(hAlpha + hBeta))/qt
		hInf = 1/(1+exp((v-(-67))/7.1))
	UNITSON
}
