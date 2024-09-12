"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %122 = "func.call"() <{callee = @inputs}> : () -> tensor<8xf32>
    %123 = "func.call"() <{callee = @expected}> : () -> tensor<i64>
    %124 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %125 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %126 = "stablehlo.shift_right_logical"(%124, %125) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %127 = "stablehlo.convert"(%126) : (tensor<i64>) -> tensor<ui32>
    %128 = "stablehlo.broadcast_in_dim"(%127) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %129 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %130 = "stablehlo.convert"(%129) : (tensor<ui32>) -> tensor<i64>
    %131 = "stablehlo.and"(%124, %130) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %132 = "stablehlo.convert"(%131) : (tensor<i64>) -> tensor<ui32>
    %133 = "stablehlo.broadcast_in_dim"(%132) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %134 = "stablehlo.concatenate"(%128, %133) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %135 = "func.call"(%134) <{callee = @_gumbel}> : (tensor<2xui32>) -> tensor<8xf32>
    %136 = "stablehlo.add"(%135, %122) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %137 = "func.call"(%136) <{callee = @argmax}> : (tensor<8xf32>) -> tensor<i64>
    "stablehlo.custom_call"(%137, %123) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i64>, tensor<i64>) -> ()
    "func.return"(%137) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %121 = "stablehlo.constant"() <{value = dense<[5.61341476, -3.17508101, 2.33323073, 0.483267874, 1.09662056, -0.71605432, 5.598730e+00, 3.95920396]> : tensor<8xf32>}> : () -> tensor<8xf32>
    "func.return"(%121) : (tensor<8xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %120 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    "func.return"(%120) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<8xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_gumbel", sym_visibility = "private"}> ({
  ^bb0(%arg34: tensor<2xui32>):
    %113 = "stablehlo.constant"() <{value = dense<1.17549435E-38> : tensor<f32>}> : () -> tensor<f32>
    %114 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %115 = "func.call"(%arg34, %113, %114) <{callee = @_uniform}> : (tensor<2xui32>, tensor<f32>, tensor<f64>) -> tensor<8xf32>
    %116 = "stablehlo.log"(%115) : (tensor<8xf32>) -> tensor<8xf32>
    %117 = "stablehlo.negate"(%116) : (tensor<8xf32>) -> tensor<8xf32>
    %118 = "stablehlo.log"(%117) : (tensor<8xf32>) -> tensor<8xf32>
    %119 = "stablehlo.negate"(%118) : (tensor<8xf32>) -> tensor<8xf32>
    "func.return"(%119) : (tensor<8xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<f32>, tensor<f64>) -> tensor<8xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform", sym_visibility = "private"}> ({
  ^bb0(%arg13: tensor<2xui32>, %arg14: tensor<f32>, %arg15: tensor<f64>):
    %68 = "stablehlo.convert"(%arg15) : (tensor<f64>) -> tensor<f32>
    %69 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %70 = "stablehlo.broadcast_in_dim"(%68) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %71 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<8xui32>
    %72 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %73 = "stablehlo.reshape"(%72) : (tensor<1xui32>) -> tensor<ui32>
    %74 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %75 = "stablehlo.reshape"(%74) : (tensor<1xui32>) -> tensor<ui32>
    %76 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 4>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<8xui32>) -> tensor<4xui32>
    %77 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 8>, start_indices = array<i64: 4>, strides = array<i64: 1>}> : (tensor<8xui32>) -> tensor<4xui32>
    %78 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %79 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %80 = "stablehlo.xor"(%73, %75) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %81 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %82 = "stablehlo.xor"(%80, %81) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %83 = "stablehlo.broadcast_in_dim"(%73) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %84 = "stablehlo.add"(%76, %83) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %85 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %86 = "stablehlo.add"(%77, %85) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %87 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %88 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %89:9 = "stablehlo.while"(%88, %87, %84, %86, %75, %82, %73, %78, %79) ({
    ^bb0(%arg25: tensor<i64>, %arg26: tensor<i64>, %arg27: tensor<4xui32>, %arg28: tensor<4xui32>, %arg29: tensor<ui32>, %arg30: tensor<ui32>, %arg31: tensor<ui32>, %arg32: tensor<4xui32>, %arg33: tensor<4xui32>):
      %111 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %112 = "stablehlo.compare"(%arg25, %111) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%112) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg16: tensor<i64>, %arg17: tensor<i64>, %arg18: tensor<4xui32>, %arg19: tensor<4xui32>, %arg20: tensor<ui32>, %arg21: tensor<ui32>, %arg22: tensor<ui32>, %arg23: tensor<4xui32>, %arg24: tensor<4xui32>):
      %108:8 = "func.call"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24) <{callee = @None}> : (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %109 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %110 = "stablehlo.add"(%arg16, %109) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%110, %108#0, %108#1, %108#2, %108#3, %108#4, %108#5, %108#6, %108#7) : (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %90 = "stablehlo.concatenate"(%89#2, %89#3) <{dimension = 0 : i64}> : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32>
    %91 = "stablehlo.constant"() <{value = dense<9> : tensor<ui32>}> : () -> tensor<ui32>
    %92 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %93 = "stablehlo.shift_right_logical"(%90, %92) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %94 = "stablehlo.constant"() <{value = dense<1065353216> : tensor<ui32>}> : () -> tensor<ui32>
    %95 = "stablehlo.broadcast_in_dim"(%94) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %96 = "stablehlo.or"(%93, %95) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %97 = "stablehlo.bitcast_convert"(%96) : (tensor<8xui32>) -> tensor<8xf32>
    %98 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %99 = "stablehlo.broadcast_in_dim"(%98) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<8xf32>
    %100 = "stablehlo.subtract"(%97, %99) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %101 = "stablehlo.subtract"(%70, %69) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %102 = "stablehlo.broadcast_in_dim"(%101) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf32>) -> tensor<8xf32>
    %103 = "stablehlo.multiply"(%100, %102) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %104 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf32>) -> tensor<8xf32>
    %105 = "stablehlo.add"(%103, %104) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %106 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf32>) -> tensor<8xf32>
    %107 = "stablehlo.maximum"(%106, %105) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    "func.return"(%107) : (tensor<8xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg5: tensor<i64>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>, %arg8: tensor<ui32>, %arg9: tensor<ui32>, %arg10: tensor<ui32>, %arg11: tensor<4xui32>, %arg12: tensor<4xui32>):
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %14 = "stablehlo.add"(%arg5, %13) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %15 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %16 = "stablehlo.reshape"(%15) : (tensor<1xui32>) -> tensor<ui32>
    %17 = "stablehlo.add"(%arg6, %arg7) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %18 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %19 = "stablehlo.shift_left"(%arg7, %18) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %20 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %21 = "stablehlo.subtract"(%20, %16) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %23 = "stablehlo.shift_right_logical"(%arg7, %22) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %24 = "stablehlo.or"(%19, %23) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %25 = "stablehlo.xor"(%17, %24) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %26 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %27 = "stablehlo.reshape"(%26) : (tensor<1xui32>) -> tensor<ui32>
    %28 = "stablehlo.add"(%17, %25) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %29 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %30 = "stablehlo.shift_left"(%25, %29) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %31 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %32 = "stablehlo.subtract"(%31, %27) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %33 = "stablehlo.broadcast_in_dim"(%32) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %34 = "stablehlo.shift_right_logical"(%25, %33) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %35 = "stablehlo.or"(%30, %34) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %36 = "stablehlo.xor"(%28, %35) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %37 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %38 = "stablehlo.reshape"(%37) : (tensor<1xui32>) -> tensor<ui32>
    %39 = "stablehlo.add"(%28, %36) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %40 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %41 = "stablehlo.shift_left"(%36, %40) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %42 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %43 = "stablehlo.subtract"(%42, %38) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %45 = "stablehlo.shift_right_logical"(%36, %44) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %46 = "stablehlo.or"(%41, %45) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %47 = "stablehlo.xor"(%39, %46) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %48 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %49 = "stablehlo.reshape"(%48) : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.add"(%39, %47) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %51 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %52 = "stablehlo.shift_left"(%47, %51) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %53 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %54 = "stablehlo.subtract"(%53, %49) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %56 = "stablehlo.shift_right_logical"(%47, %55) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %57 = "stablehlo.or"(%52, %56) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %58 = "stablehlo.xor"(%50, %57) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %59 = "stablehlo.broadcast_in_dim"(%arg8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %60 = "stablehlo.add"(%50, %59) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %61 = "stablehlo.broadcast_in_dim"(%arg9) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %62 = "stablehlo.add"(%58, %61) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %63 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %64 = "stablehlo.add"(%arg5, %63) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %65 = "stablehlo.convert"(%64) : (tensor<i64>) -> tensor<ui32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %67 = "stablehlo.add"(%62, %66) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    "func.return"(%14, %60, %67, %arg9, %arg10, %arg8, %arg12, %arg11) : (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8xf32>) -> tensor<i64>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8xf32>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<8xi64>
    %1 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<i64>, %arg3: tensor<f32>, %arg4: tensor<i64>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%11, %12) : (tensor<f32>, tensor<i64>) -> ()
    }) : (tensor<8xf32>, tensor<8xi64>, tensor<f32>, tensor<i64>) -> (tensor<f32>, tensor<i64>)
    "func.return"(%3#1) : (tensor<i64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

