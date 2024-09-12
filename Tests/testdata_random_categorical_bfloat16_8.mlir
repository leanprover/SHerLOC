"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %137 = "func.call"() <{callee = @inputs}> : () -> tensor<8xbf16>
    %138 = "func.call"() <{callee = @expected}> : () -> tensor<i64>
    %139 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %140 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %141 = "stablehlo.shift_right_logical"(%139, %140) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %142 = "stablehlo.convert"(%141) : (tensor<i64>) -> tensor<ui32>
    %143 = "stablehlo.broadcast_in_dim"(%142) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %144 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %145 = "stablehlo.convert"(%144) : (tensor<ui32>) -> tensor<i64>
    %146 = "stablehlo.and"(%139, %145) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %147 = "stablehlo.convert"(%146) : (tensor<i64>) -> tensor<ui32>
    %148 = "stablehlo.broadcast_in_dim"(%147) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %149 = "stablehlo.concatenate"(%143, %148) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %150 = "func.call"(%149) <{callee = @_gumbel}> : (tensor<2xui32>) -> tensor<8xbf16>
    %151 = "stablehlo.add"(%150, %137) : (tensor<8xbf16>, tensor<8xbf16>) -> tensor<8xbf16>
    %152 = "func.call"(%151) <{callee = @argmax}> : (tensor<8xbf16>) -> tensor<i64>
    "stablehlo.custom_call"(%152, %138) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i64>, tensor<i64>) -> ()
    "func.return"(%152) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %136 = "stablehlo.constant"() <{value = dense<[3.093750e+00, -1.039060e+00, 5.468750e-01, 1.375000e+00, 2.890630e+00, 1.789060e+00, -5.390630e-01, -3.984380e+00]> : tensor<8xbf16>}> : () -> tensor<8xbf16>
    "func.return"(%136) : (tensor<8xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %135 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    "func.return"(%135) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<8xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_gumbel", sym_visibility = "private"}> ({
  ^bb0(%arg34: tensor<2xui32>):
    %128 = "stablehlo.constant"() <{value = dense<1.175490e-38> : tensor<bf16>}> : () -> tensor<bf16>
    %129 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %130 = "func.call"(%arg34, %128, %129) <{callee = @_uniform}> : (tensor<2xui32>, tensor<bf16>, tensor<f64>) -> tensor<8xbf16>
    %131 = "stablehlo.log"(%130) : (tensor<8xbf16>) -> tensor<8xbf16>
    %132 = "stablehlo.negate"(%131) : (tensor<8xbf16>) -> tensor<8xbf16>
    %133 = "stablehlo.log"(%132) : (tensor<8xbf16>) -> tensor<8xbf16>
    %134 = "stablehlo.negate"(%133) : (tensor<8xbf16>) -> tensor<8xbf16>
    "func.return"(%134) : (tensor<8xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<bf16>, tensor<f64>) -> tensor<8xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform", sym_visibility = "private"}> ({
  ^bb0(%arg13: tensor<2xui32>, %arg14: tensor<bf16>, %arg15: tensor<f64>):
    %68 = "stablehlo.convert"(%arg15) : (tensor<f64>) -> tensor<bf16>
    %69 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<1xbf16>
    %70 = "stablehlo.broadcast_in_dim"(%68) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<1xbf16>
    %71 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xui32>
    %72 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %73 = "stablehlo.reshape"(%72) : (tensor<1xui32>) -> tensor<ui32>
    %74 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %75 = "stablehlo.reshape"(%74) : (tensor<1xui32>) -> tensor<ui32>
    %76 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %77 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %78 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %79 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %80 = "stablehlo.xor"(%73, %75) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %81 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %82 = "stablehlo.xor"(%80, %81) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %83 = "stablehlo.broadcast_in_dim"(%73) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %84 = "stablehlo.add"(%76, %83) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %85 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %86 = "stablehlo.add"(%77, %85) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %87 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %88 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %89:9 = "stablehlo.while"(%88, %87, %84, %86, %75, %82, %73, %78, %79) ({
    ^bb0(%arg25: tensor<i64>, %arg26: tensor<i64>, %arg27: tensor<1xui32>, %arg28: tensor<1xui32>, %arg29: tensor<ui32>, %arg30: tensor<ui32>, %arg31: tensor<ui32>, %arg32: tensor<4xui32>, %arg33: tensor<4xui32>):
      %126 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %127 = "stablehlo.compare"(%arg25, %126) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%127) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg16: tensor<i64>, %arg17: tensor<i64>, %arg18: tensor<1xui32>, %arg19: tensor<1xui32>, %arg20: tensor<ui32>, %arg21: tensor<ui32>, %arg22: tensor<ui32>, %arg23: tensor<4xui32>, %arg24: tensor<4xui32>):
      %123:8 = "func.call"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24) <{callee = @None}> : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %124 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %125 = "stablehlo.add"(%arg16, %124) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%125, %123#0, %123#1, %123#2, %123#3, %123#4, %123#5, %123#6, %123#7) : (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %90 = "stablehlo.concatenate"(%89#2, %89#3) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %91 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1x2xui32>
    %92 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4x1xui32>
    %93 = "stablehlo.constant"() <{value = dense<8> : tensor<ui32>}> : () -> tensor<ui32>
    %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4x1xui32>
    %95 = "stablehlo.multiply"(%94, %92) : (tensor<4x1xui32>, tensor<4x1xui32>) -> tensor<4x1xui32>
    %96 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x2xui32>) -> tensor<4x2xui32>
    %97 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<4x1xui32>) -> tensor<4x2xui32>
    %98 = "stablehlo.shift_right_logical"(%96, %97) : (tensor<4x2xui32>, tensor<4x2xui32>) -> tensor<4x2xui32>
    %99 = "stablehlo.constant"() <{value = dense<255> : tensor<ui32>}> : () -> tensor<ui32>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4x2xui32>
    %101 = "stablehlo.and"(%100, %98) : (tensor<4x2xui32>, tensor<4x2xui32>) -> tensor<4x2xui32>
    %102 = "stablehlo.transpose"(%101) <{permutation = array<i64: 1, 0>}> : (tensor<4x2xui32>) -> tensor<2x4xui32>
    %103 = "stablehlo.reshape"(%102) : (tensor<2x4xui32>) -> tensor<8xui32>
    %104 = "stablehlo.convert"(%103) : (tensor<8xui32>) -> tensor<8xui8>
    %105 = "stablehlo.convert"(%104) : (tensor<8xui8>) -> tensor<8xui16>
    %106 = "stablehlo.constant"() <{value = dense<1> : tensor<ui16>}> : () -> tensor<ui16>
    %107 = "stablehlo.broadcast_in_dim"(%106) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<8xui16>
    %108 = "stablehlo.shift_right_logical"(%105, %107) : (tensor<8xui16>, tensor<8xui16>) -> tensor<8xui16>
    %109 = "stablehlo.constant"() <{value = dense<16256> : tensor<ui16>}> : () -> tensor<ui16>
    %110 = "stablehlo.broadcast_in_dim"(%109) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<8xui16>
    %111 = "stablehlo.or"(%108, %110) : (tensor<8xui16>, tensor<8xui16>) -> tensor<8xui16>
    %112 = "stablehlo.bitcast_convert"(%111) : (tensor<8xui16>) -> tensor<8xbf16>
    %113 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %114 = "stablehlo.broadcast_in_dim"(%113) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<8xbf16>
    %115 = "stablehlo.subtract"(%112, %114) : (tensor<8xbf16>, tensor<8xbf16>) -> tensor<8xbf16>
    %116 = "stablehlo.subtract"(%70, %69) : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    %117 = "stablehlo.broadcast_in_dim"(%116) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xbf16>) -> tensor<8xbf16>
    %118 = "stablehlo.multiply"(%115, %117) : (tensor<8xbf16>, tensor<8xbf16>) -> tensor<8xbf16>
    %119 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xbf16>) -> tensor<8xbf16>
    %120 = "stablehlo.add"(%118, %119) : (tensor<8xbf16>, tensor<8xbf16>) -> tensor<8xbf16>
    %121 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xbf16>) -> tensor<8xbf16>
    %122 = "stablehlo.maximum"(%121, %120) : (tensor<8xbf16>, tensor<8xbf16>) -> tensor<8xbf16>
    "func.return"(%122) : (tensor<8xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg5: tensor<i64>, %arg6: tensor<1xui32>, %arg7: tensor<1xui32>, %arg8: tensor<ui32>, %arg9: tensor<ui32>, %arg10: tensor<ui32>, %arg11: tensor<4xui32>, %arg12: tensor<4xui32>):
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %14 = "stablehlo.add"(%arg5, %13) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %15 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %16 = "stablehlo.reshape"(%15) : (tensor<1xui32>) -> tensor<ui32>
    %17 = "stablehlo.add"(%arg6, %arg7) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %18 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %19 = "stablehlo.shift_left"(%arg7, %18) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %20 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %21 = "stablehlo.subtract"(%20, %16) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %23 = "stablehlo.shift_right_logical"(%arg7, %22) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %24 = "stablehlo.or"(%19, %23) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %25 = "stablehlo.xor"(%17, %24) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %26 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %27 = "stablehlo.reshape"(%26) : (tensor<1xui32>) -> tensor<ui32>
    %28 = "stablehlo.add"(%17, %25) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %29 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %30 = "stablehlo.shift_left"(%25, %29) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %31 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %32 = "stablehlo.subtract"(%31, %27) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %33 = "stablehlo.broadcast_in_dim"(%32) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %34 = "stablehlo.shift_right_logical"(%25, %33) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %35 = "stablehlo.or"(%30, %34) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %36 = "stablehlo.xor"(%28, %35) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %37 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %38 = "stablehlo.reshape"(%37) : (tensor<1xui32>) -> tensor<ui32>
    %39 = "stablehlo.add"(%28, %36) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %40 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %41 = "stablehlo.shift_left"(%36, %40) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %42 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %43 = "stablehlo.subtract"(%42, %38) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %45 = "stablehlo.shift_right_logical"(%36, %44) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %46 = "stablehlo.or"(%41, %45) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %47 = "stablehlo.xor"(%39, %46) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %48 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %49 = "stablehlo.reshape"(%48) : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.add"(%39, %47) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %51 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %52 = "stablehlo.shift_left"(%47, %51) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %53 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %54 = "stablehlo.subtract"(%53, %49) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %56 = "stablehlo.shift_right_logical"(%47, %55) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %57 = "stablehlo.or"(%52, %56) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %58 = "stablehlo.xor"(%50, %57) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %59 = "stablehlo.broadcast_in_dim"(%arg8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %60 = "stablehlo.add"(%50, %59) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %61 = "stablehlo.broadcast_in_dim"(%arg9) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %62 = "stablehlo.add"(%58, %61) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %63 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %64 = "stablehlo.add"(%arg5, %63) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %65 = "stablehlo.convert"(%64) : (tensor<i64>) -> tensor<ui32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %67 = "stablehlo.add"(%62, %66) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    "func.return"(%14, %60, %67, %arg9, %arg10, %arg8, %arg12, %arg11) : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8xbf16>) -> tensor<i64>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8xbf16>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<8xi64>
    %1 = "stablehlo.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<i64>, %arg3: tensor<bf16>, %arg4: tensor<i64>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%11, %12) : (tensor<bf16>, tensor<i64>) -> ()
    }) : (tensor<8xbf16>, tensor<8xi64>, tensor<bf16>, tensor<i64>) -> (tensor<bf16>, tensor<i64>)
    "func.return"(%3#1) : (tensor<i64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

