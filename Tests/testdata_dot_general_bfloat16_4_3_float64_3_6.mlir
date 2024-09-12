"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xbf16>, tensor<3x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xbf16>) -> tensor<4x3xf64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf64>) -> tensor<3x6xf64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    "func.return"(%7) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xbf16>, tensor<3x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.828130e+00, -3.421880e+00, 1.781250e+00], [-3.796880e+00, -3.808590e-02, 3.710940e-01], [-6.132810e-01, 1.000000e+00, 1.806640e-01], [-6.562500e+00, 1.062500e+00, 3.796880e+00]]> : tensor<4x3xbf16>}> : () -> tensor<4x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-0.050378914882338442, -3.7698173417195675, -3.4866203881372089, -2.8050004607341901, 2.4681323215395947, -0.39923405596434125], [5.0350009816030941, 2.7373967085451238, 0.90214104352151581, -0.46061932072390066, 4.5521023701500845, -1.3741391936675766], [3.7719597840306021, -0.35389950656538899, -2.7256187914652052, 1.9568224855047538, -1.0825140636812025, 0.45062046791383625]]> : tensor<3x6xf64>}> : () -> tensor<3x6xf64>
    "func.return"(%1, %2) : (tensor<4x3xbf16>, tensor<3x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-10.367862749966964, 0.6641018114282069, 1.9185759296029601, 12.994663718421322, -24.485140195643627, 6.6338840763269129], [1.3992704108512108, 14.077939004372627, 12.192442800413428, 11.393943837240791, -9.9462751980279833, 1.7353996249508321], [5.7473550036653691, 4.9854180775737289, 2.5479985897413444, 1.6131623679083755, 2.8428717063728501, -1.0478855084042311], [20.001959976859826, 26.304198118873394, 13.490637307047594, 25.348217869949838, -15.470680177358691, 2.8709001881045366]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%0) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

