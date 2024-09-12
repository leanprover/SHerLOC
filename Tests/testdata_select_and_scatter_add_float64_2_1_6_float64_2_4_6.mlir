"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x1x6xf64>, tensor<2x4x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xf64>
    %5 = "stablehlo.constant"() <{value = dense<0xFFF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xf64>, tensor<f64>) -> tensor<2x4x6xf64>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%10) : (tensor<f64>) -> ()
    }) : (tensor<2x4x6xf64>, tensor<2x1x6xf64>, tensor<f64>) -> tensor<2x4x6xf64>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xf64>) -> tensor<2x4x6xf64>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xf64>, tensor<2x4x6xf64>) -> ()
    "func.return"(%9) : (tensor<2x4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x1x6xf64>, tensor<2x4x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[0.65534158846581381, 2.0149102770534553, -3.9497793154591596, -0.84784950026977868, 0.20737941184148967, 1.8757323675827537]], [[5.7686491291416271, 4.0626762668865215, -1.0563197582057429, 0.4078860017863033, 1.1773181971925688, 7.7005518869319491]]]> : tensor<2x1x6xf64>}> : () -> tensor<2x1x6xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[[-0.91188340903326082, -2.4710970674799979, -0.92104401692248761, -2.3088683218007087, -3.0882282943931707, -0.19730095004274356], [5.0314787758858932, -1.0738800580670875, 0.38513681322200699, -2.1599522948671259, 2.0200116940160111, -2.2082048968279562], [-7.8978149272775831, -2.4943593800822326, -2.8310341747127472, 2.2036443618103254, -2.1779128165069794, 0.62039528728551752], [-1.2715074440429766, 1.0804422535410538, -1.9581270259674506, 1.6861035708929502, 2.686715461937089, 0.61568725222015164]], [[-3.3825742798177849, 3.3532386739088937, -3.3167885166654756, 5.005068848558377, -0.18242536091999154, 0.37390408014017096], [-1.404168952083892, 1.632259753969096, 1.4609210279428617, 1.8578002827028763, 0.14578291434060806, -0.22041549769900609], [-0.43491736596543723, 2.605963479901976, 1.5871054686508652, 0.6376353756926213, -2.0242079062777498, 1.999748887433044], [2.2884318173121345, -0.96852848645466061, -0.14375187992771277, -3.6057230073502269, -3.2691972473950717, -0.64468238524284938]]]> : tensor<2x4x6xf64>}> : () -> tensor<2x4x6xf64>
    "func.return"(%1, %2) : (tensor<2x1x6xf64>, tensor<2x4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.65534158846581381, 2.0149102770534553, -3.9497793154591596, 0.000000e+00, 0.20737941184148967, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -0.84784950026977868, 0.000000e+00, 1.8757323675827537], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 4.0626762668865215, 0.000000e+00, 0.4078860017863033, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.1773181971925688, 0.000000e+00], [5.7686491291416271, 0.000000e+00, -1.0563197582057429, 0.000000e+00, 0.000000e+00, 7.7005518869319491], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf64>}> : () -> tensor<2x4x6xf64>
    "func.return"(%0) : (tensor<2x4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

