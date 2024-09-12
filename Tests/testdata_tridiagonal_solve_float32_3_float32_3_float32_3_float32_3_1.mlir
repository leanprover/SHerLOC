"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x1xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %32:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>)
    %33 = "func.call"() <{callee = @expected}> : () -> tensor<3x1xf32>
    %34 = "stablehlo.transpose"(%32#3) <{permutation = array<i64: 1, 0>}> : (tensor<3x1xf32>) -> tensor<1x3xf32>
    %35 = "stablehlo.transpose"(%34) <{permutation = array<i64: 1, 0>}> : (tensor<1x3xf32>) -> tensor<3x1xf32>
    %36 = "stablehlo.slice"(%32#2) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %37 = "stablehlo.reshape"(%36) : (tensor<1xf32>) -> tensor<f32>
    %38 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %39 = "stablehlo.reshape"(%38) : (tensor<1xf32>) -> tensor<f32>
    %40 = "stablehlo.divide"(%37, %39) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %41 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<0xf32>
    %42 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<3xf32>
    %43 = "stablehlo.slice"(%32#2) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<0xf32>
    %44 = "stablehlo.slice"(%32#2) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<3xf32>
    %45 = "stablehlo.slice"(%32#0) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<0xf32>
    %46 = "stablehlo.slice"(%32#0) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<3xf32>
    %47 = "stablehlo.reshape"(%41) : (tensor<0xf32>) -> tensor<0x32xf32>
    %48 = "stablehlo.reshape"(%43) : (tensor<0xf32>) -> tensor<0x32xf32>
    %49 = "stablehlo.reshape"(%45) : (tensor<0xf32>) -> tensor<0x32xf32>
    %50 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %51 = "stablehlo.broadcast_in_dim"(%50) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<0x32xf32>
    %52 = "stablehlo.reshape"(%51) : (tensor<0x32xf32>) -> tensor<0xf32>
    %53 = "stablehlo.slice"(%42) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %54 = "stablehlo.reshape"(%53) : (tensor<1xf32>) -> tensor<f32>
    %55 = "stablehlo.slice"(%44) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %56 = "stablehlo.reshape"(%55) : (tensor<1xf32>) -> tensor<f32>
    %57 = "stablehlo.slice"(%46) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %58 = "stablehlo.reshape"(%57) : (tensor<1xf32>) -> tensor<f32>
    %59:2 = "func.call"(%40, %54, %56, %58) <{callee = @None}> : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %60 = "stablehlo.slice"(%42) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %61 = "stablehlo.reshape"(%60) : (tensor<1xf32>) -> tensor<f32>
    %62 = "stablehlo.slice"(%44) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %63 = "stablehlo.reshape"(%62) : (tensor<1xf32>) -> tensor<f32>
    %64 = "stablehlo.slice"(%46) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %65 = "stablehlo.reshape"(%64) : (tensor<1xf32>) -> tensor<f32>
    %66:2 = "func.call"(%59#0, %61, %63, %65) <{callee = @None}> : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %67 = "stablehlo.slice"(%42) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %68 = "stablehlo.reshape"(%67) : (tensor<1xf32>) -> tensor<f32>
    %69 = "stablehlo.slice"(%44) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %70 = "stablehlo.reshape"(%69) : (tensor<1xf32>) -> tensor<f32>
    %71 = "stablehlo.slice"(%46) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %72 = "stablehlo.reshape"(%71) : (tensor<1xf32>) -> tensor<f32>
    %73:2 = "func.call"(%66#0, %68, %70, %72) <{callee = @None}> : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    %74 = "stablehlo.broadcast_in_dim"(%59#1) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %75 = "stablehlo.broadcast_in_dim"(%66#1) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %76 = "stablehlo.broadcast_in_dim"(%73#1) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %77 = "stablehlo.concatenate"(%74, %75, %76) <{dimension = 0 : i64}> : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %78 = "stablehlo.concatenate"(%52, %77) <{dimension = 0 : i64}> : (tensor<0xf32>, tensor<3xf32>) -> tensor<3xf32>
    %79 = "stablehlo.slice"(%35) <{limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %80 = "stablehlo.reshape"(%79) : (tensor<1x1xf32>) -> tensor<1xf32>
    %81 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %82 = "stablehlo.divide"(%80, %81) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %83 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %84 = "stablehlo.broadcast_in_dim"(%83) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %85 = "stablehlo.slice"(%78) <{limit_indices = array<i64: 2>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<2xf32>
    %86 = "func.call"(%84, %85) <{callee = @append}> : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
    %87 = "stablehlo.slice"(%35) <{limit_indices = array<i64: 0, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<0x1xf32>
    %88 = "stablehlo.slice"(%35) <{limit_indices = array<i64: 3, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %89 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<0xf32>
    %90 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<3xf32>
    %91 = "stablehlo.slice"(%86) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<0xf32>
    %92 = "stablehlo.slice"(%86) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<3xf32>
    %93 = "stablehlo.slice"(%32#0) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<0xf32>
    %94 = "stablehlo.slice"(%32#0) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<3xf32>
    %95 = "stablehlo.reshape"(%87) : (tensor<0x1xf32>) -> tensor<0x32x1xf32>
    %96 = "stablehlo.reshape"(%89) : (tensor<0xf32>) -> tensor<0x32xf32>
    %97 = "stablehlo.reshape"(%91) : (tensor<0xf32>) -> tensor<0x32xf32>
    %98 = "stablehlo.reshape"(%93) : (tensor<0xf32>) -> tensor<0x32xf32>
    %99 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<0x32x1xf32>
    %101 = "stablehlo.reshape"(%100) : (tensor<0x32x1xf32>) -> tensor<0x1xf32>
    %102 = "stablehlo.slice"(%88) <{limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %103 = "stablehlo.reshape"(%102) : (tensor<1x1xf32>) -> tensor<1xf32>
    %104 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %105 = "stablehlo.reshape"(%104) : (tensor<1xf32>) -> tensor<f32>
    %106 = "stablehlo.slice"(%92) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %107 = "stablehlo.reshape"(%106) : (tensor<1xf32>) -> tensor<f32>
    %108 = "stablehlo.slice"(%94) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %109 = "stablehlo.reshape"(%108) : (tensor<1xf32>) -> tensor<f32>
    %110:2 = "func.call"(%82, %103, %105, %107, %109) <{callee = @None_0}> : (tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %111 = "stablehlo.slice"(%88) <{limit_indices = array<i64: 2, 1>, start_indices = array<i64: 1, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %112 = "stablehlo.reshape"(%111) : (tensor<1x1xf32>) -> tensor<1xf32>
    %113 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %114 = "stablehlo.reshape"(%113) : (tensor<1xf32>) -> tensor<f32>
    %115 = "stablehlo.slice"(%92) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %116 = "stablehlo.reshape"(%115) : (tensor<1xf32>) -> tensor<f32>
    %117 = "stablehlo.slice"(%94) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %118 = "stablehlo.reshape"(%117) : (tensor<1xf32>) -> tensor<f32>
    %119:2 = "func.call"(%110#0, %112, %114, %116, %118) <{callee = @None_0}> : (tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %120 = "stablehlo.slice"(%88) <{limit_indices = array<i64: 3, 1>, start_indices = array<i64: 2, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %121 = "stablehlo.reshape"(%120) : (tensor<1x1xf32>) -> tensor<1xf32>
    %122 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %123 = "stablehlo.reshape"(%122) : (tensor<1xf32>) -> tensor<f32>
    %124 = "stablehlo.slice"(%92) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %125 = "stablehlo.reshape"(%124) : (tensor<1xf32>) -> tensor<f32>
    %126 = "stablehlo.slice"(%94) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %127 = "stablehlo.reshape"(%126) : (tensor<1xf32>) -> tensor<f32>
    %128:2 = "func.call"(%119#0, %121, %123, %125, %127) <{callee = @None_0}> : (tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %129 = "stablehlo.broadcast_in_dim"(%110#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf32>) -> tensor<1x1xf32>
    %130 = "stablehlo.broadcast_in_dim"(%119#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf32>) -> tensor<1x1xf32>
    %131 = "stablehlo.broadcast_in_dim"(%128#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf32>) -> tensor<1x1xf32>
    %132 = "stablehlo.concatenate"(%129, %130, %131) <{dimension = 0 : i64}> : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<3x1xf32>
    %133 = "stablehlo.concatenate"(%101, %132) <{dimension = 0 : i64}> : (tensor<0x1xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
    %134 = "stablehlo.constant"() <{value = dense<-1> : tensor<i64>}> : () -> tensor<i64>
    %135 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %136 = "stablehlo.compare"(%134, %135) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %137 = "stablehlo.constant"() <{value = dense<-1> : tensor<i64>}> : () -> tensor<i64>
    %138 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %139 = "stablehlo.add"(%137, %138) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %140 = "stablehlo.constant"() <{value = dense<-1> : tensor<i64>}> : () -> tensor<i64>
    %141 = "stablehlo.select"(%136, %139, %140) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %142 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %143 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %144 = "stablehlo.compare"(%142, %143) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %145 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %146 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %147 = "stablehlo.add"(%145, %146) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %148 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %149 = "stablehlo.select"(%144, %147, %148) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %150 = "stablehlo.dynamic_slice"(%133, %141, %149) <{slice_sizes = array<i64: 1, 1>}> : (tensor<3x1xf32>, tensor<i64>, tensor<i64>) -> tensor<1x1xf32>
    %151 = "stablehlo.reshape"(%150) : (tensor<1x1xf32>) -> tensor<1xf32>
    %152 = "stablehlo.reverse"(%133) <{dimensions = array<i64: 0>}> : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %153 = "stablehlo.reverse"(%78) <{dimensions = array<i64: 0>}> : (tensor<3xf32>) -> tensor<3xf32>
    %154 = "stablehlo.slice"(%152) <{limit_indices = array<i64: 0, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<0x1xf32>
    %155 = "stablehlo.slice"(%152) <{limit_indices = array<i64: 3, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %156 = "stablehlo.slice"(%153) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<0xf32>
    %157 = "stablehlo.slice"(%153) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<3xf32>
    %158 = "stablehlo.reshape"(%154) : (tensor<0x1xf32>) -> tensor<0x32x1xf32>
    %159 = "stablehlo.reshape"(%156) : (tensor<0xf32>) -> tensor<0x32xf32>
    %160 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %161 = "stablehlo.broadcast_in_dim"(%160) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<0x32x1xf32>
    %162 = "stablehlo.reshape"(%161) : (tensor<0x32x1xf32>) -> tensor<0x1xf32>
    %163 = "stablehlo.slice"(%155) <{limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %164 = "stablehlo.reshape"(%163) : (tensor<1x1xf32>) -> tensor<1xf32>
    %165 = "stablehlo.slice"(%157) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %166 = "stablehlo.reshape"(%165) : (tensor<1xf32>) -> tensor<f32>
    %167:2 = "func.call"(%151, %164, %166) <{callee = @None_1}> : (tensor<1xf32>, tensor<1xf32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %168 = "stablehlo.slice"(%155) <{limit_indices = array<i64: 2, 1>, start_indices = array<i64: 1, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %169 = "stablehlo.reshape"(%168) : (tensor<1x1xf32>) -> tensor<1xf32>
    %170 = "stablehlo.slice"(%157) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %171 = "stablehlo.reshape"(%170) : (tensor<1xf32>) -> tensor<f32>
    %172:2 = "func.call"(%167#0, %169, %171) <{callee = @None_1}> : (tensor<1xf32>, tensor<1xf32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %173 = "stablehlo.slice"(%155) <{limit_indices = array<i64: 3, 1>, start_indices = array<i64: 2, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf32>) -> tensor<1x1xf32>
    %174 = "stablehlo.reshape"(%173) : (tensor<1x1xf32>) -> tensor<1xf32>
    %175 = "stablehlo.slice"(%157) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf32>) -> tensor<1xf32>
    %176 = "stablehlo.reshape"(%175) : (tensor<1xf32>) -> tensor<f32>
    %177:2 = "func.call"(%172#0, %174, %176) <{callee = @None_1}> : (tensor<1xf32>, tensor<1xf32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>)
    %178 = "stablehlo.broadcast_in_dim"(%167#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf32>) -> tensor<1x1xf32>
    %179 = "stablehlo.broadcast_in_dim"(%172#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf32>) -> tensor<1x1xf32>
    %180 = "stablehlo.broadcast_in_dim"(%177#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf32>) -> tensor<1x1xf32>
    %181 = "stablehlo.concatenate"(%178, %179, %180) <{dimension = 0 : i64}> : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<3x1xf32>
    %182 = "stablehlo.concatenate"(%162, %181) <{dimension = 0 : i64}> : (tensor<0x1xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
    %183 = "stablehlo.reverse"(%182) <{dimensions = array<i64: 0>}> : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %184 = "stablehlo.transpose"(%183) <{permutation = array<i64: 1, 0>}> : (tensor<3x1xf32>) -> tensor<1x3xf32>
    %185 = "stablehlo.transpose"(%184) <{permutation = array<i64: 1, 0>}> : (tensor<1x3xf32>) -> tensor<3x1xf32>
    "stablehlo.custom_call"(%185, %33) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x1xf32>, tensor<3x1xf32>) -> ()
    "func.return"(%185) : (tensor<3x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %28 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %29 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<3xf32>}> : () -> tensor<3xf32>
    %30 = "stablehlo.constant"() <{value = dense<[1.000000e+00, 2.000000e+00, 0.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %31 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
    "func.return"(%28, %29, %30, %31) : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x1xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %27 = "stablehlo.constant"() <{value = dense<[[0.571428597], [0.428571403], [-0.285714298]]> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
    "func.return"(%27) : (tensor<3x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>, %arg12: tensor<f32>, %arg13: tensor<f32>):
    %21 = "stablehlo.multiply"(%arg13, %arg10) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %22 = "stablehlo.subtract"(%arg11, %21) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %23 = "stablehlo.divide"(%arg12, %22) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %24 = "stablehlo.multiply"(%arg13, %arg10) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %25 = "stablehlo.subtract"(%arg11, %24) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %26 = "stablehlo.divide"(%arg12, %25) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%23, %26) : (tensor<f32>, tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "append", sym_visibility = "private"}> ({
  ^bb0(%arg8: tensor<1xf32>, %arg9: tensor<2xf32>):
    %20 = "stablehlo.concatenate"(%arg8, %arg9) <{dimension = 0 : i64}> : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
    "func.return"(%20) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>), sym_name = "None_0", sym_visibility = "private"}> ({
  ^bb0(%arg3: tensor<1xf32>, %arg4: tensor<1xf32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %6 = "stablehlo.broadcast_in_dim"(%arg7) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %7 = "stablehlo.multiply"(%6, %arg3) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %8 = "stablehlo.subtract"(%arg4, %7) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %9 = "stablehlo.multiply"(%arg7, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %10 = "stablehlo.subtract"(%arg5, %9) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %12 = "stablehlo.divide"(%8, %11) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %13 = "stablehlo.broadcast_in_dim"(%arg7) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %14 = "stablehlo.multiply"(%13, %arg3) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %15 = "stablehlo.subtract"(%arg4, %14) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %16 = "stablehlo.multiply"(%arg7, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %17 = "stablehlo.subtract"(%arg5, %16) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %19 = "stablehlo.divide"(%15, %18) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    "func.return"(%12, %19) : (tensor<1xf32>, tensor<1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<1xf32>, tensor<1xf32>, tensor<f32>) -> (tensor<1xf32>, tensor<1xf32>), sym_name = "None_1", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<f32>):
    %0 = "stablehlo.broadcast_in_dim"(%arg2) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %1 = "stablehlo.multiply"(%0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %2 = "stablehlo.subtract"(%arg1, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %3 = "stablehlo.broadcast_in_dim"(%arg2) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1xf32>
    %4 = "stablehlo.multiply"(%3, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %5 = "stablehlo.subtract"(%arg1, %4) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    "func.return"(%2, %5) : (tensor<1xf32>, tensor<1xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

