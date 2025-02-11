import numpy as np

def conversion(value,variable_name,dxc,dyc,drf,hfacc,time_unit="y"):
    mol_s = ['ADVxTr01','ADVyTr01','ADVrTr01','DFxETr01','DFyETr01','DFrETr01','DFrITr01']
    mol_m3_s = ['cDIC_PIC','respDIC','rDIC_DOC','rDIC_POC','dDIC_PIC']
    mol_m2_s = ['fluxCO2']
    other = ['TRAC01','gDICEpr']
    time_conversion = {"y":3600 * 24 * 365,
                       "d": 3600 * 24,
                       "h": 3600,
                       "s": 1}
    carbon_conversion = 12
    mmol_to_mol = 10**-3
    
    if variable_name in mol_s:
        #  f[x,y,z] * 10^-3 * 12 * (3600 * 24 * 365)
        # print("mol_s ran")
        output = value * mmol_to_mol * carbon_conversion * time_conversion[time_unit]
        
    elif variable_name in mol_m3_s:
        # f[x,y,z] * 10^-3 * DXC[x,y] * DYC[x,y] * DRF[z] * hFacC[x,y,z] * 12 * (3600 * 24 * 365)
        output = value * mmol_to_mol * \
                 dxc * dyc * drf[:, np.newaxis, np.newaxis] * hfacc * \
                 carbon_conversion * \
                 time_conversion[time_unit]
        
    elif variable_name in mol_m2_s:
        # f[x,y,z] * 10^-3 * DXC[x,y] * DYC[x,y] * hFacC[x,y,0] * 12 * (3600 * 24 * 365)
        output = value * mmol_to_mol * \
                 dxc * dyc * hfacc[0] * \
                 carbon_conversion * \
                 time_conversion[time_unit]
        
    elif variable_name == "gDICEpr":
        # f[x,y,z] * 10^-3 * DXC[x,y] * DYC[x,y] * DRF[z] * hFacC[x,y,0] * 12 * (3600 * 24 * 365)
        output = value * mmol_to_mol * \
                 dxc * dyc * drf[0] * hfacc[0] *\
                 carbon_conversion * \
                 time_conversion[time_unit]
        
    elif variable_name == "TRAC01":
        # f[x,y,z] * 10^-3
        output = value * mmol_to_mol

    else: "not found"
        
    return output