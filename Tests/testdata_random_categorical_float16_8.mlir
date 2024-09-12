"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %136 = "func.call"() <{callee = @inputs}> : () -> tensor<8xf16>
    %137 = "func.call"() <{callee = @expected}> : () -> tensor<i64>
    %138 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %139 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %140 = "stablehlo.shift_right_logical"(%138, %139) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %141 = "stablehlo.convert"(%140) : (tensor<i64>) -> tensor<ui32>
    %142 = "stablehlo.broadcast_in_dim"(%141) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %143 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %144 = "stablehlo.convert"(%143) : (tensor<ui32>) -> tensor<i64>
    %145 = "stablehlo.and"(%138, %144) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %146 = "stablehlo.convert"(%145) : (tensor<i64>) -> tensor<ui32>
    %147 = "stablehlo.broadcast_in_dim"(%146) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %148 = "stablehlo.concatenate"(%142, %147) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %149 = "func.call"(%148) <{callee = @_gumbel}> : (tensor<2xui32>) -> tensor<8xf16>
    %150 = "stablehlo.add"(%149, %136) : (tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
    %151 = "func.call"(%150) <{callee = @argmax}> : (tensor<8xf16>) -> tensor<i64>
    "stablehlo.custom_call"(%151, %137) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i64>, tensor<i64>) -> ()
    "func.return"(%151) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %135 = "stablehlo.constant"() <{value = dense<[-5.035160e+00, 5.683590e-01, 2.695310e+00, -2.638670e+00, -4.597660e+00, -1.592770e+00, -1.658200e+00, 1.249020e+00]> : tensor<8xf16>}> : () -> tensor<8xf16>
    "func.return"(%135) : (tensor<8xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %134 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    "func.return"(%134) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<8xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_gumbel", sym_visibility = "private"}> ({
  ^bb0(%arg34: tensor<2xui32>):
    %127 = "stablehlo.constant"() <{value = dense<6.103520e-05> : tensor<f16>}> : () -> tensor<f16>
    %128 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %129 = "func.call"(%arg34, %127, %128) <{callee = @_uniform}> : (tensor<2xui32>, tensor<f16>, tensor<f64>) -> tensor<8xf16>
    %130 = "stablehlo.log"(%129) : (tensor<8xf16>) -> tensor<8xf16>
    %131 = "stablehlo.negate"(%130) : (tensor<8xf16>) -> tensor<8xf16>
    %132 = "stablehlo.log"(%131) : (tensor<8xf16>) -> tensor<8xf16>
    %133 = "stablehlo.negate"(%132) : (tensor<8xf16>) -> tensor<8xf16>
    "func.return"(%133) : (tensor<8xf16>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<f16>, tensor<f64>) -> tensor<8xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform", sym_visibility = "private"}> ({
  ^bb0(%arg13: tensor<2xui32>, %arg14: tensor<f16>, %arg15: tensor<f64>):
    %68 = "stablehlo.convert"(%arg15) : (tensor<f64>) -> tensor<f16>
    %69 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<f16>) -> tensor<1xf16>
    %70 = "stablehlo.broadcast_in_dim"(%68) <{broadcast_dimensions = array<i64>}> : (tensor<f16>) -> tensor<1xf16>
    %71 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xui32>
    %72 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %73 = "stablehlo.reshape"(%72) : (tensor<1xui32>) -> tensor<ui32>
    %74 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %75 = "stablehlo.reshape"(%74) : (tensor<1xui32>) -> tensor<ui32>
    %76 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 2>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<2xui32>
    %77 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 4>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<2xui32>
    %78 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %79 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %80 = "stablehlo.xor"(%73, %75) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %81 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %82 = "stablehlo.xor"(%80, %81) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %83 = "stablehlo.broadcast_in_dim"(%73) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %84 = "stablehlo.add"(%76, %83) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %85 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %86 = "stablehlo.add"(%77, %85) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %87 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %88 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %89:9 = "stablehlo.while"(%88, %87, %84, %86, %75, %82, %73, %78, %79) ({
    ^bb0(%arg25: tensor<i64>, %arg26: tensor<i64>, %arg27: tensor<2xui32>, %arg28: tensor<2xui32>, %arg29: tensor<ui32>, %arg30: tensor<ui32>, %arg31: tensor<ui32>, %arg32: tensor<4xui32>, %arg33: tensor<4xui32>):
      %125 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %126 = "stablehlo.compare"(%arg25, %125) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%126) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg16: tensor<i64>, %arg17: tensor<i64>, %arg18: tensor<2xui32>, %arg19: tensor<2xui32>, %arg20: tensor<ui32>, %arg21: tensor<ui32>, %arg22: tensor<ui32>, %arg23: tensor<4xui32>, %arg24: tensor<4xui32>):
      %122:8 = "func.call"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24) <{callee = @None}> : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %123 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %124 = "stablehlo.add"(%arg16, %123) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%124, %122#0, %122#1, %122#2, %122#3, %122#4, %122#5, %122#6, %122#7) : (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %90 = "stablehlo.concatenate"(%89#2, %89#3) <{dimension = 0 : i64}> : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %91 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1x4xui32>
    %92 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2x1xui32>
    %93 = "stablehlo.constant"() <{value = dense<16> : tensor<ui32>}> : () -> tensor<ui32>
    %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2x1xui32>
    %95 = "stablehlo.multiply"(%94, %92) : (tensor<2x1xui32>, tensor<2x1xui32>) -> tensor<2x1xui32>
    %96 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x4xui32>) -> tensor<2x4xui32>
    %97 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<2x1xui32>) -> tensor<2x4xui32>
    %98 = "stablehlo.shift_right_logical"(%96, %97) : (tensor<2x4xui32>, tensor<2x4xui32>) -> tensor<2x4xui32>
    %99 = "stablehlo.constant"() <{value = dense<65535> : tensor<ui32>}> : () -> tensor<ui32>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2x4xui32>
    %101 = "stablehlo.and"(%100, %98) : (tensor<2x4xui32>, tensor<2x4xui32>) -> tensor<2x4xui32>
    %102 = "stablehlo.transpose"(%101) <{permutation = array<i64: 1, 0>}> : (tensor<2x4xui32>) -> tensor<4x2xui32>
    %103 = "stablehlo.reshape"(%102) : (tensor<4x2xui32>) -> tensor<8xui32>
    %104 = "stablehlo.convert"(%103) : (tensor<8xui32>) -> tensor<8xui16>
    %105 = "stablehlo.constant"() <{value = dense<6> : tensor<ui16>}> : () -> tensor<ui16>
    %106 = "stablehlo.broadcast_in_dim"(%105) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<8xui16>
    %107 = "stablehlo.shift_right_logical"(%104, %106) : (tensor<8xui16>, tensor<8xui16>) -> tensor<8xui16>
    %108 = "stablehlo.constant"() <{value = dense<15360> : tensor<ui16>}> : () -> tensor<ui16>
    %109 = "stablehlo.broadcast_in_dim"(%108) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<8xui16>
    %110 = "stablehlo.or"(%107, %109) : (tensor<8xui16>, tensor<8xui16>) -> tensor<8xui16>
    %111 = "stablehlo.bitcast_convert"(%110) : (tensor<8xui16>) -> tensor<8xf16>
    %112 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %113 = "stablehlo.broadcast_in_dim"(%112) <{broadcast_dimensions = array<i64>}> : (tensor<f16>) -> tensor<8xf16>
    %114 = "stablehlo.subtract"(%111, %113) : (tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
    %115 = "stablehlo.subtract"(%70, %69) : (tensor<1xf16>, tensor<1xf16>) -> tensor<1xf16>
    %116 = "stablehlo.broadcast_in_dim"(%115) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf16>) -> tensor<8xf16>
    %117 = "stablehlo.multiply"(%114, %116) : (tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
    %118 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf16>) -> tensor<8xf16>
    %119 = "stablehlo.add"(%117, %118) : (tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
    %120 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf16>) -> tensor<8xf16>
    %121 = "stablehlo.maximum"(%120, %119) : (tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
    "func.return"(%121) : (tensor<8xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg5: tensor<i64>, %arg6: tensor<2xui32>, %arg7: tensor<2xui32>, %arg8: tensor<ui32>, %arg9: tensor<ui32>, %arg10: tensor<ui32>, %arg11: tensor<4xui32>, %arg12: tensor<4xui32>):
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %14 = "stablehlo.add"(%arg5, %13) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %15 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %16 = "stablehlo.reshape"(%15) : (tensor<1xui32>) -> tensor<ui32>
    %17 = "stablehlo.add"(%arg6, %arg7) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %18 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %19 = "stablehlo.shift_left"(%arg7, %18) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %20 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %21 = "stablehlo.subtract"(%20, %16) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %23 = "stablehlo.shift_right_logical"(%arg7, %22) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %24 = "stablehlo.or"(%19, %23) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %25 = "stablehlo.xor"(%17, %24) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %26 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %27 = "stablehlo.reshape"(%26) : (tensor<1xui32>) -> tensor<ui32>
    %28 = "stablehlo.add"(%17, %25) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %29 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %30 = "stablehlo.shift_left"(%25, %29) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %31 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %32 = "stablehlo.subtract"(%31, %27) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %33 = "stablehlo.broadcast_in_dim"(%32) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %34 = "stablehlo.shift_right_logical"(%25, %33) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %35 = "stablehlo.or"(%30, %34) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %36 = "stablehlo.xor"(%28, %35) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %37 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %38 = "stablehlo.reshape"(%37) : (tensor<1xui32>) -> tensor<ui32>
    %39 = "stablehlo.add"(%28, %36) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %40 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %41 = "stablehlo.shift_left"(%36, %40) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %42 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %43 = "stablehlo.subtract"(%42, %38) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %45 = "stablehlo.shift_right_logical"(%36, %44) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %46 = "stablehlo.or"(%41, %45) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %47 = "stablehlo.xor"(%39, %46) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %48 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %49 = "stablehlo.reshape"(%48) : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.add"(%39, %47) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %51 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %52 = "stablehlo.shift_left"(%47, %51) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %53 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %54 = "stablehlo.subtract"(%53, %49) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %56 = "stablehlo.shift_right_logical"(%47, %55) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %57 = "stablehlo.or"(%52, %56) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %58 = "stablehlo.xor"(%50, %57) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %59 = "stablehlo.broadcast_in_dim"(%arg8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %60 = "stablehlo.add"(%50, %59) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %61 = "stablehlo.broadcast_in_dim"(%arg9) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %62 = "stablehlo.add"(%58, %61) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %63 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %64 = "stablehlo.add"(%arg5, %63) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %65 = "stablehlo.convert"(%64) : (tensor<i64>) -> tensor<ui32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %67 = "stablehlo.add"(%62, %66) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "func.return"(%14, %60, %67, %arg9, %arg10, %arg8, %arg12, %arg11) : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8xf16>) -> tensor<i64>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8xf16>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<8xi64>
    %1 = "stablehlo.constant"() <{value = dense<0xFC00> : tensor<f16>}> : () -> tensor<f16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<i64>, %arg3: tensor<f16>, %arg4: tensor<i64>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%11, %12) : (tensor<f16>, tensor<i64>) -> ()
    }) : (tensor<8xf16>, tensor<8xi64>, tensor<f16>, tensor<i64>) -> (tensor<f16>, tensor<i64>)
    "func.return"(%3#1) : (tensor<i64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

