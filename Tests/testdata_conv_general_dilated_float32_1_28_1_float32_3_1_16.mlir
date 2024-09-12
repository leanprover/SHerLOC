"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x28x16xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x28x1xf32>, tensor<3x1x16xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<1x28x16xf32>
    %5 = "stablehlo.convolution"(%3#0, %3#1) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<1x2xi64>}> : (tensor<1x28x1xf32>, tensor<3x1x16xf32>) -> tensor<1x28x16xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<1x28x16xf32>, tensor<1x28x16xf32>) -> ()
    "func.return"(%5) : (tensor<1x28x16xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x28x1xf32>, tensor<3x1x16xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-0.23315914], [-2.94537687], [2.53514838], [0.497039199], [-0.0617677122], [-0.301088601], [0.146781266], [-2.38990903], [4.86341095], [-5.245030e+00], [0.296509117], [-0.992398321], [-0.233479097], [-0.0227273535], [5.03361654], [-6.31293392], [-1.99352169], [-0.145096213], [-9.53260517], [0.304931343], [3.99902487], [-0.607606649], [5.70198345], [1.85295963], [2.39537239], [0.510458767], [5.8290453], [1.22793019]]]> : tensor<1x28x1xf32>}> : () -> tensor<1x28x1xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[3.44075513, -1.71694124, 3.07970071, -3.5951736, 3.29311776, -1.00775051, -0.99696207, -6.87695932, -1.08986819, -1.60310304, -1.67757082, -1.06905067, -1.23166823, 2.04847169, -1.69546521, 1.31356537]], [[-4.14485931, 5.65509796, 1.40218759, -1.07793689, 3.83094072, -0.34269011, -6.15694618, -1.68585372, -3.09812593, -4.144280e+00, -2.71217704, -3.79190016, -3.22533059, 4.05559349, 5.42157745, -5.86706114]], [[-2.23933268, 3.65689158, -0.896553516, 0.510010958, -0.17592001, 1.84497321, -3.88200188, 1.07630432, 2.36944056, -5.07903624, -1.79240155, -2.18313432, -0.653279125, -0.985935866, 3.00582767, -4.48060322]]]> : tensor<3x1x16xf32>}> : () -> tensor<3x1x16xf32>
    "func.return"(%1, %2) : (tensor<1x28x1xf32>, tensor<3x1x16xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x28x16xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xA5FCF1406F6E41C190141440A51BA0BFED08C0BEEF55ABC080E94D412EBB31C09135C8C0B4D07E41602CBD407F0EEA4058462B4058ABFA3FD0E021C1570A69410F53B740AD87DFC0A3DEE3C0D4CBA94038F547C1BC7DBD408F68084187C21441A62D7641CE7B97BE2A797540B543BC40E117024102C26EC1707FFEC0B3B1B340A00AAEC17AB0A941A3C6BEC02DC20141926999BDA80D41402EA169C16921844104DA5DC015F204C16BD634C07D94F1C01AF59BC070827040C1DBA14141C2A7C1ADA1D9406546E2BF04F50841BEE71AC1723724419CB335C01022ABC04EB592C108608EC099EDB9C005B0AFC0F5B88EC05FED95C0C1A2E840EF03E5BF6ED0303F12FD2840717013C07366DB3F94DCEFBFBF00BA3F3E8284BF4BDC863FD0D568C01D2988BFEF087D3F23A901BEB268B83E33765DBE3142883F824905C07651174096ED343FD9A987BF2E773EBF53191F3FBFFBB0BF115ADF3E5A3BAC3F7B908B3F9988AC3F54EF193FF539283F4725633F4B88733FF904BFBFD3108BBFB28B833FB9466D406190ECC0FAEAB53F65DA96BEB2CD0FBC90FF84C009C90A411BC83FBF5542B9C0763440416E808C4007739F40B6B6BA3F8A6D15408713BCC0973917410AB8F5BEAC918040BB4CE8C06BEC9040E27118C1964D1A4156F889C0F410044184249641478470C1F5D11EC01026DBBFA0358B4045FD62C141D1B43F6872F2C0481685C176D04641512C8540E2B62C3F3AF13A4160F60EC18B65E6C0E8D32540DE1FC7C1D30C25415E4A613EF5F38DC0990E15C139FF9F4199756A41E0C102C12B3D17424CB513C29F70EB4058E03AC11E2784C0E39F23C07659D2419E45C2C1F76D3A4112F34641EE20B1405AAC6041F4BA2B41B39F39C1CE290FC2C4540F42A86D88C106B3E140DD8F6DC1923F904165647FC1909956402231E840E9010A42D5901C40D984434142601C41F4C7D44016DEC4406A0309C1018EF0405DD985C04301B5404133DFC0FABF89BE3C35ECBD123232C08D6AC7BE9F11D740D1091EBF61AF0C4066589A4062352740F72B7D40AB3D3F40BAFA4BC05DB7D2C02242E840895719C017D2993E3B4057C01BB57340AF1285C0BDE2843FE0F72040E033E6406323E03F0A222B40C3AE1540AE78FF3FE0BEFE3FDA453DC05E66B23E5A1F2C3E86B23FC1197095413871A8C014975B4001E8DEBFBC7A1841AC5799C1FEF8E140150744418FC7C8C1651A09C1D6732AC1E85B3BC08F10B1C0F5717641DED0B5C127C3D9C0A367AD40105E4A41A40509C1C08DA241939655C18FCACEC011FD71C19C39F4C155D43341971C13C06FFBA8C0DF5341C133BCD4413CA60541DA68A3BF9ECC3F42F6874EC2A5FF06419DEF44C14D3DE8C0B3CAD2C00F5B264290F3D0C18094154124BFE141060344418745B7410D6C77413F3755C1A10243C283545242B62252C1F81877BF73DBB0C1F92AC641C834E3C1D6DFD840560C994141793A4204674B4198F39841D80E82415FFF6941C8CC644164FDA6C151720ABF91B88140F66B7141A80701C298070D40F38C1D407333AEC0947578C1AE8B1F42686A6C40C9B79FC1B7DA5042AB97A64147F0BB41A5691241C73E9740077BD0C128C7234220511942A42C52C24A6361C14E3E2F41843314C2416F7E40509C66423C2C8B41984FF3412FC218425F68CC415D8B0E426ECDF54170081DC2A4134AC26D7C5942D8122CC282DC0242B50F02C2CBED0F42296BF7C16A0A874198BCFCC0DDB08A42335C97419961C9C03FE5FF40E8BA9B3E04520241F4DDB1C1B3AFEE4149EA00C22AA662C17BF49E415FEBE2408CF0B6C0E07283418A1E33C05489B4C144E217C1879562C11E9D5FC1E84B24C14D9D62C16F074EC179898B4188B39A415CB7A2C1339B604037CA28410741CB40C5062DC1576A1D41B957D640420CB3C1C2B7A2C17D8D3041D16903C2CA7E74C1C0B566C16E1AD6C077DED83DB912E24005D885C1D3FDEEC120422042A8CE8E40D81441C0F4229C41F8EC0440E6C626C2EA265CC0ECCD49C16F4500C245228EC1EB22C8C1C4D296C14A6DA041EC0D1642B3372AC21065D2404C2C174192169041DF33AAC1CBA2CB41481BFBBFF822D3C12B081FC202F1C8C0CFE3E7C1771397C199CF92C1FD0669C10CAB86414983F240D1D361C1894596C08DB3434103B9094123BC0FC118057341528ADFBFEC9D94C13FDA81C1C6B203C1EFD777C14C5228C167DA42C1147725C1221D50411A123641F9825EC17DAADDC0AAB8A041A1783740790DC6C0E6170D4128A602414D46E1C12CF430C15AE91941EC3E0EC2169D7DC199C689C1A27906C1AE709D3F91D181417ABACFC14C3BC9C1AB4F124228500A41B4C0EFC09E5DBE4167637CBECFA824C24A4040C18A4B7BC19EB2F9C1F5EE96C106A3CAC11ADAA1C149CDBB4160B60942731F1CC2B0776F419A1944C056639D41913DB2C1EA32BF41D470C9C03AF255C1F89F28C2C88322C166EF66C177BE51C128342EC1283D32C16E5D8741A2704EC0DDACE73E"> : tensor<1x28x16xf32>}> : () -> tensor<1x28x16xf32>
    "func.return"(%0) : (tensor<1x28x16xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

