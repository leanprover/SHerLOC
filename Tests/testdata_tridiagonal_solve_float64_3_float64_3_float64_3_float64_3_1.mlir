"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x1xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %32:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3x1xf64>)
    %33 = "func.call"() <{callee = @expected}> : () -> tensor<3x1xf64>
    %34 = "stablehlo.transpose"(%32#3) <{permutation = array<i64: 1, 0>}> : (tensor<3x1xf64>) -> tensor<1x3xf64>
    %35 = "stablehlo.transpose"(%34) <{permutation = array<i64: 1, 0>}> : (tensor<1x3xf64>) -> tensor<3x1xf64>
    %36 = "stablehlo.slice"(%32#2) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %37 = "stablehlo.reshape"(%36) : (tensor<1xf64>) -> tensor<f64>
    %38 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %39 = "stablehlo.reshape"(%38) : (tensor<1xf64>) -> tensor<f64>
    %40 = "stablehlo.divide"(%37, %39) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %41 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<0xf64>
    %42 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<3xf64>
    %43 = "stablehlo.slice"(%32#2) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<0xf64>
    %44 = "stablehlo.slice"(%32#2) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<3xf64>
    %45 = "stablehlo.slice"(%32#0) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<0xf64>
    %46 = "stablehlo.slice"(%32#0) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<3xf64>
    %47 = "stablehlo.reshape"(%41) : (tensor<0xf64>) -> tensor<0x32xf64>
    %48 = "stablehlo.reshape"(%43) : (tensor<0xf64>) -> tensor<0x32xf64>
    %49 = "stablehlo.reshape"(%45) : (tensor<0xf64>) -> tensor<0x32xf64>
    %50 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %51 = "stablehlo.broadcast_in_dim"(%50) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<0x32xf64>
    %52 = "stablehlo.reshape"(%51) : (tensor<0x32xf64>) -> tensor<0xf64>
    %53 = "stablehlo.slice"(%42) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %54 = "stablehlo.reshape"(%53) : (tensor<1xf64>) -> tensor<f64>
    %55 = "stablehlo.slice"(%44) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %56 = "stablehlo.reshape"(%55) : (tensor<1xf64>) -> tensor<f64>
    %57 = "stablehlo.slice"(%46) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %58 = "stablehlo.reshape"(%57) : (tensor<1xf64>) -> tensor<f64>
    %59:2 = "func.call"(%40, %54, %56, %58) <{callee = @None}> : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %60 = "stablehlo.slice"(%42) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %61 = "stablehlo.reshape"(%60) : (tensor<1xf64>) -> tensor<f64>
    %62 = "stablehlo.slice"(%44) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %63 = "stablehlo.reshape"(%62) : (tensor<1xf64>) -> tensor<f64>
    %64 = "stablehlo.slice"(%46) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %65 = "stablehlo.reshape"(%64) : (tensor<1xf64>) -> tensor<f64>
    %66:2 = "func.call"(%59#0, %61, %63, %65) <{callee = @None}> : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %67 = "stablehlo.slice"(%42) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %68 = "stablehlo.reshape"(%67) : (tensor<1xf64>) -> tensor<f64>
    %69 = "stablehlo.slice"(%44) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %70 = "stablehlo.reshape"(%69) : (tensor<1xf64>) -> tensor<f64>
    %71 = "stablehlo.slice"(%46) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %72 = "stablehlo.reshape"(%71) : (tensor<1xf64>) -> tensor<f64>
    %73:2 = "func.call"(%66#0, %68, %70, %72) <{callee = @None}> : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %74 = "stablehlo.broadcast_in_dim"(%59#1) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %75 = "stablehlo.broadcast_in_dim"(%66#1) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %76 = "stablehlo.broadcast_in_dim"(%73#1) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %77 = "stablehlo.concatenate"(%74, %75, %76) <{dimension = 0 : i64}> : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %78 = "stablehlo.concatenate"(%52, %77) <{dimension = 0 : i64}> : (tensor<0xf64>, tensor<3xf64>) -> tensor<3xf64>
    %79 = "stablehlo.slice"(%35) <{limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %80 = "stablehlo.reshape"(%79) : (tensor<1x1xf64>) -> tensor<1xf64>
    %81 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %82 = "stablehlo.divide"(%80, %81) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %83 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %84 = "stablehlo.broadcast_in_dim"(%83) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %85 = "stablehlo.slice"(%78) <{limit_indices = array<i64: 2>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<2xf64>
    %86 = "func.call"(%84, %85) <{callee = @append}> : (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>
    %87 = "stablehlo.slice"(%35) <{limit_indices = array<i64: 0, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<0x1xf64>
    %88 = "stablehlo.slice"(%35) <{limit_indices = array<i64: 3, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %89 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<0xf64>
    %90 = "stablehlo.slice"(%32#1) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<3xf64>
    %91 = "stablehlo.slice"(%86) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<0xf64>
    %92 = "stablehlo.slice"(%86) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<3xf64>
    %93 = "stablehlo.slice"(%32#0) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<0xf64>
    %94 = "stablehlo.slice"(%32#0) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<3xf64>
    %95 = "stablehlo.reshape"(%87) : (tensor<0x1xf64>) -> tensor<0x32x1xf64>
    %96 = "stablehlo.reshape"(%89) : (tensor<0xf64>) -> tensor<0x32xf64>
    %97 = "stablehlo.reshape"(%91) : (tensor<0xf64>) -> tensor<0x32xf64>
    %98 = "stablehlo.reshape"(%93) : (tensor<0xf64>) -> tensor<0x32xf64>
    %99 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<0x32x1xf64>
    %101 = "stablehlo.reshape"(%100) : (tensor<0x32x1xf64>) -> tensor<0x1xf64>
    %102 = "stablehlo.slice"(%88) <{limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %103 = "stablehlo.reshape"(%102) : (tensor<1x1xf64>) -> tensor<1xf64>
    %104 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %105 = "stablehlo.reshape"(%104) : (tensor<1xf64>) -> tensor<f64>
    %106 = "stablehlo.slice"(%92) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %107 = "stablehlo.reshape"(%106) : (tensor<1xf64>) -> tensor<f64>
    %108 = "stablehlo.slice"(%94) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %109 = "stablehlo.reshape"(%108) : (tensor<1xf64>) -> tensor<f64>
    %110:2 = "func.call"(%82, %103, %105, %107, %109) <{callee = @None_0}> : (tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %111 = "stablehlo.slice"(%88) <{limit_indices = array<i64: 2, 1>, start_indices = array<i64: 1, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %112 = "stablehlo.reshape"(%111) : (tensor<1x1xf64>) -> tensor<1xf64>
    %113 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %114 = "stablehlo.reshape"(%113) : (tensor<1xf64>) -> tensor<f64>
    %115 = "stablehlo.slice"(%92) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %116 = "stablehlo.reshape"(%115) : (tensor<1xf64>) -> tensor<f64>
    %117 = "stablehlo.slice"(%94) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %118 = "stablehlo.reshape"(%117) : (tensor<1xf64>) -> tensor<f64>
    %119:2 = "func.call"(%110#0, %112, %114, %116, %118) <{callee = @None_0}> : (tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %120 = "stablehlo.slice"(%88) <{limit_indices = array<i64: 3, 1>, start_indices = array<i64: 2, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %121 = "stablehlo.reshape"(%120) : (tensor<1x1xf64>) -> tensor<1xf64>
    %122 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %123 = "stablehlo.reshape"(%122) : (tensor<1xf64>) -> tensor<f64>
    %124 = "stablehlo.slice"(%92) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %125 = "stablehlo.reshape"(%124) : (tensor<1xf64>) -> tensor<f64>
    %126 = "stablehlo.slice"(%94) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %127 = "stablehlo.reshape"(%126) : (tensor<1xf64>) -> tensor<f64>
    %128:2 = "func.call"(%119#0, %121, %123, %125, %127) <{callee = @None_0}> : (tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %129 = "stablehlo.broadcast_in_dim"(%110#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf64>) -> tensor<1x1xf64>
    %130 = "stablehlo.broadcast_in_dim"(%119#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf64>) -> tensor<1x1xf64>
    %131 = "stablehlo.broadcast_in_dim"(%128#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf64>) -> tensor<1x1xf64>
    %132 = "stablehlo.concatenate"(%129, %130, %131) <{dimension = 0 : i64}> : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<3x1xf64>
    %133 = "stablehlo.concatenate"(%101, %132) <{dimension = 0 : i64}> : (tensor<0x1xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
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
    %150 = "stablehlo.dynamic_slice"(%133, %141, %149) <{slice_sizes = array<i64: 1, 1>}> : (tensor<3x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
    %151 = "stablehlo.reshape"(%150) : (tensor<1x1xf64>) -> tensor<1xf64>
    %152 = "stablehlo.reverse"(%133) <{dimensions = array<i64: 0>}> : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %153 = "stablehlo.reverse"(%78) <{dimensions = array<i64: 0>}> : (tensor<3xf64>) -> tensor<3xf64>
    %154 = "stablehlo.slice"(%152) <{limit_indices = array<i64: 0, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<0x1xf64>
    %155 = "stablehlo.slice"(%152) <{limit_indices = array<i64: 3, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %156 = "stablehlo.slice"(%153) <{limit_indices = array<i64: 0>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<0xf64>
    %157 = "stablehlo.slice"(%153) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<3xf64>
    %158 = "stablehlo.reshape"(%154) : (tensor<0x1xf64>) -> tensor<0x32x1xf64>
    %159 = "stablehlo.reshape"(%156) : (tensor<0xf64>) -> tensor<0x32xf64>
    %160 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %161 = "stablehlo.broadcast_in_dim"(%160) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<0x32x1xf64>
    %162 = "stablehlo.reshape"(%161) : (tensor<0x32x1xf64>) -> tensor<0x1xf64>
    %163 = "stablehlo.slice"(%155) <{limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %164 = "stablehlo.reshape"(%163) : (tensor<1x1xf64>) -> tensor<1xf64>
    %165 = "stablehlo.slice"(%157) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %166 = "stablehlo.reshape"(%165) : (tensor<1xf64>) -> tensor<f64>
    %167:2 = "func.call"(%151, %164, %166) <{callee = @None_1}> : (tensor<1xf64>, tensor<1xf64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %168 = "stablehlo.slice"(%155) <{limit_indices = array<i64: 2, 1>, start_indices = array<i64: 1, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %169 = "stablehlo.reshape"(%168) : (tensor<1x1xf64>) -> tensor<1xf64>
    %170 = "stablehlo.slice"(%157) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %171 = "stablehlo.reshape"(%170) : (tensor<1xf64>) -> tensor<f64>
    %172:2 = "func.call"(%167#0, %169, %171) <{callee = @None_1}> : (tensor<1xf64>, tensor<1xf64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %173 = "stablehlo.slice"(%155) <{limit_indices = array<i64: 3, 1>, start_indices = array<i64: 2, 0>, strides = array<i64: 1, 1>}> : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %174 = "stablehlo.reshape"(%173) : (tensor<1x1xf64>) -> tensor<1xf64>
    %175 = "stablehlo.slice"(%157) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    %176 = "stablehlo.reshape"(%175) : (tensor<1xf64>) -> tensor<f64>
    %177:2 = "func.call"(%172#0, %174, %176) <{callee = @None_1}> : (tensor<1xf64>, tensor<1xf64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>)
    %178 = "stablehlo.broadcast_in_dim"(%167#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf64>) -> tensor<1x1xf64>
    %179 = "stablehlo.broadcast_in_dim"(%172#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf64>) -> tensor<1x1xf64>
    %180 = "stablehlo.broadcast_in_dim"(%177#1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xf64>) -> tensor<1x1xf64>
    %181 = "stablehlo.concatenate"(%178, %179, %180) <{dimension = 0 : i64}> : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<3x1xf64>
    %182 = "stablehlo.concatenate"(%162, %181) <{dimension = 0 : i64}> : (tensor<0x1xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %183 = "stablehlo.reverse"(%182) <{dimensions = array<i64: 0>}> : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %184 = "stablehlo.transpose"(%183) <{permutation = array<i64: 1, 0>}> : (tensor<3x1xf64>) -> tensor<1x3xf64>
    %185 = "stablehlo.transpose"(%184) <{permutation = array<i64: 1, 0>}> : (tensor<1x3xf64>) -> tensor<3x1xf64>
    "stablehlo.custom_call"(%185, %33) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x1xf64>, tensor<3x1xf64>) -> ()
    "func.return"(%185) : (tensor<3x1xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3x1xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %28 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %29 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<3xf64>}> : () -> tensor<3xf64>
    %30 = "stablehlo.constant"() <{value = dense<[1.000000e+00, 2.000000e+00, 0.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %31 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<3x1xf64>}> : () -> tensor<3x1xf64>
    "func.return"(%28, %29, %30, %31) : (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3x1xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x1xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %27 = "stablehlo.constant"() <{value = dense<[[0.5714285714285714], [0.4285714285714286], [-0.2857142857142857]]> : tensor<3x1xf64>}> : () -> tensor<3x1xf64>
    "func.return"(%27) : (tensor<3x1xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg10: tensor<f64>, %arg11: tensor<f64>, %arg12: tensor<f64>, %arg13: tensor<f64>):
    %21 = "stablehlo.multiply"(%arg13, %arg10) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %22 = "stablehlo.subtract"(%arg11, %21) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %23 = "stablehlo.divide"(%arg12, %22) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %24 = "stablehlo.multiply"(%arg13, %arg10) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %25 = "stablehlo.subtract"(%arg11, %24) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %26 = "stablehlo.divide"(%arg12, %25) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "func.return"(%23, %26) : (tensor<f64>, tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "append", sym_visibility = "private"}> ({
  ^bb0(%arg8: tensor<1xf64>, %arg9: tensor<2xf64>):
    %20 = "stablehlo.concatenate"(%arg8, %arg9) <{dimension = 0 : i64}> : (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>
    "func.return"(%20) : (tensor<3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>), sym_name = "None_0", sym_visibility = "private"}> ({
  ^bb0(%arg3: tensor<1xf64>, %arg4: tensor<1xf64>, %arg5: tensor<f64>, %arg6: tensor<f64>, %arg7: tensor<f64>):
    %6 = "stablehlo.broadcast_in_dim"(%arg7) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %7 = "stablehlo.multiply"(%6, %arg3) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %8 = "stablehlo.subtract"(%arg4, %7) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %9 = "stablehlo.multiply"(%arg7, %arg6) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %10 = "stablehlo.subtract"(%arg5, %9) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %12 = "stablehlo.divide"(%8, %11) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %13 = "stablehlo.broadcast_in_dim"(%arg7) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %14 = "stablehlo.multiply"(%13, %arg3) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %15 = "stablehlo.subtract"(%arg4, %14) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %16 = "stablehlo.multiply"(%arg7, %arg6) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %17 = "stablehlo.subtract"(%arg5, %16) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %18 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %19 = "stablehlo.divide"(%15, %18) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    "func.return"(%12, %19) : (tensor<1xf64>, tensor<1xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<1xf64>, tensor<1xf64>, tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>), sym_name = "None_1", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<f64>):
    %0 = "stablehlo.broadcast_in_dim"(%arg2) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %1 = "stablehlo.multiply"(%0, %arg0) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %2 = "stablehlo.subtract"(%arg1, %1) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %3 = "stablehlo.broadcast_in_dim"(%arg2) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %4 = "stablehlo.multiply"(%3, %arg0) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %5 = "stablehlo.subtract"(%arg1, %4) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    "func.return"(%2, %5) : (tensor<1xf64>, tensor<1xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

